from pathlib import Path
import requests
from pdf2image import convert_from_path
from PIL import Image
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

from langchain_community.document_loaders import AsyncHtmlLoader
import html2text
from urllib.parse import urlparse

from pyzotero import zotero

class Document():
    def __init__(self,data, zot, chunk_max = 20000):
        self.zot = zot
        self.data = data
        self.chunk_max = chunk_max

    @staticmethod
    def load(url, data, zot=None):

        parsed_url = urlparse(url)
        if parsed_url.netloc == "arxiv.org":
            return Arxiv(data, zot)
        elif parsed_url.netloc == "github.com":
            return Github(data, zot)
        elif parsed_url.netloc == "x.com":
            return X(data, zot)
        elif data['data']['itemType'] == 'attachment' and data['data']['contentType'] == 'application/pdf':
            if not zot:
                raise Exception("Need Zotero for PDF download.")
            return Pdf(data, zot)
        elif data['data']['itemType'] == 'book' and data['data']['links'].get("attachment") and data['data']['links'].get("attachment")['attachmentType'] == 'application/pdf':
            if not zot:
                raise Exception("Need Zotero for indexing PDF books.")
            return Book(data, zot)
        else:
            return Document(data, zot)

    @staticmethod
    def download_pdf(pdf_url, folder="temp_folder"):
        response = requests.get(pdf_url, stream=True)

        file_name = pdf_url.split("/")[-1]
        if response.status_code == 200:
            output_filename = f"{folder}/{file_name}.pdf"
            chunk_size = 2000
            with open(output_filename, 'wb') as fd:
                for chunk in response.iter_content(chunk_size):
                    fd.write(chunk)
            print(f"PDF downloaded successfully to {output_filename}")
            return output_filename
        else:
            print(f"Failed to download PDF. Status code: {response.status_code}")
            return None

    @staticmethod
    def split_by_top_level_headings(markdown_text, level = 1):
        lines = markdown_text.splitlines()
        sections = []
        current_section = []
        start_str = ('#'*level)+ ' '

        for line in lines:
            if line.strip().startswith(start_str):  # Only top-level headings
                if current_section:
                    sections.append('\n'.join(current_section).strip())
                    current_section = []
            current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section).strip())
        
        return sections

    def get_metadata(self):
        metadata = {'url': self.data['data'].get('url',''), 'title': self.data['data'].get('title',''), 'abstractNote': self.data['data'].get('abstractNote',''), 'accessDate' : self.data['data'].get('accessDate','')}
        metadata['itemType'] = self.data['data'].get('itemType')
        metadata['creators'] = [ creator.get('firstName', '') + " " +  creator.get('lastName', '') for creator in self.data['data'].get('creators', [])]
        return metadata

    def get_text(self):
        if not self.data['data'].get('url'):
            return []
        url = self.data['data']['url']
        link = url
        loader = AsyncHtmlLoader([link])
        docs = loader.load()

        page_text = html2text.html2text(docs[0].page_content,bodywidth=1000)
        metadata = self.get_metadata()

        if len(page_text) < self.chunk_max:
            context = f"The following page is from the web page {url} with title '{metadata['title']}'. "
            if metadata['creators']:
                context += f"It was created by the author(s) {','.join(metadata['creators'])}."
            context += "\n"
            result = [(context + page_text, metadata)]
        else:
            chunk_sizes=np.array([self.chunk_max + 100])
            chunks = []
            level = 1
            while(level <= 3 and np.quantile(chunk_sizes, 0.80) > self.chunk_max): 
                chunks = Document.split_by_top_level_headings(page_text, level = level)
                chunk_sizes=np.array([len(chunk) for chunk in chunks])
                level += 1

            if level == 4:
                print(f"Made it to level 3 in chunking following document: ",metadata)
            # Make sure each chunk is at most chunk size:
            chunks = [chunk[:self.chunk_max] for chunk in chunks]

            # Add basic context info to each chunk.
            context = "The following is chunk {i} "+  f" from the web page  {url} with title '{metadata['title']}'. "
            if metadata['creators']:
                context += f"It was created by the author(s) {','.join(metadata['creators'])}."
            context += "\n"
            result = [(context.replace("{i}", str(i)) + chunk, {"page": i,**metadata, "fulltext": context.replace("{i}", str(i)) + chunk}) for i, chunk in enumerate(chunks)] # TODO: process metadata for qdrant payload

        return result

    def get_images(self):
        return []

class Github(Document):
    pass

class X(Document):
    def get_text(self):
        return []

class Arxiv(Document):

    def get_text(self):
        return []

    @staticmethod
    def get_base64(img):
        im_file = BytesIO()
        img.save(im_file, format="PNG")  # Specify the format of your image (e.g., "JPEG", "PNG")

        # 3. Get the byte data
        im_bytes = im_file.getvalue()

        # 4. Encode to base64
        im_b64_bytes = base64.b64encode(im_bytes)

        # 5. Decode to UTF-8 string
        return im_b64_bytes.decode('utf-8')
    
    @staticmethod
    def get_img(base6_img):
        im_bytes = base64.b64decode(base6_img.encode())   # im_bytes is a binary image
        im_file = BytesIO(im_bytes)  # convert image to file-like object
        return Image.open(im_file)

    def get_images(self):
        if not self.data['data'].get('url'):
            return []
        pdf_url = self.data['data']['url'].replace("/abs/", "/pdf/")
        path = Document.download_pdf(pdf_url)
        images = convert_from_path(
            path,
            thread_count=os.cpu_count() - 1,
            output_folder=Path(path).parents[0],
            paths_only=True,
            fmt="png"
        )
        metadata = self.get_metadata()
        metadata['date'] = self.data['data'].get('date')

        images = [Image.open(image_path) for image_path in images]
        # Add context info.
        for i,image in enumerate(images):
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default(size=24)

            context = "The following is page {i} from the arXiv paper\n"+  f"'{metadata['title']}'. "
            if metadata['creators']:
                context += f"\nIt was created by the author(s) {', '.join(metadata['creators'][:3])} on {metadata['date']}."
            text_color = (0, 255, 0) # White color
            text_position = (15, 15) # X, Y coordinates

            draw.text(text_position, context.replace("{i}", str(i+1)), fill=text_color,font=font)

        metadata_images = [{**metadata, "page": i, "image_b64": Arxiv.get_base64(images[i])} for i in range(len(images))]
        return [(image, meta) for image, meta in zip(images,metadata_images)]  # TODO: process metadata for qdrant payload
    
class Pdf(Document):

    def get_text(self):
        return []
    
    def get_images(self):
        file_name = f"{self.data['key']}.pdf"
        self.zot.dump(self.data['key'], file_name, "temp_folder")
        path = f"temp_folder/{file_name}"
        images = convert_from_path(
            path,
            thread_count=os.cpu_count() - 1,
            output_folder=Path(path).parents[0],
            paths_only=True,
            fmt="png"
        )
        metadata = self.get_metadata() #{ 'title': self.data['data']['title'], 'accessDate' : self.data['data']['dateAdded']}
        
        images = [Image.open(image_path) for image_path in images]
        metadata_images = [{**metadata, "page": i, "image_b64": Arxiv.get_base64(images[i])} for i in range(len(images))]
        return [(image, meta) for image, meta in zip(images,metadata_images)]  # TODO: process metadata for qdrant payload
    
class Book(Document):

    def get_text(self):
        return []
    
    def get_images(self):
        key = self.data['links'].get("attachment")['href'].split("/")[-1]
        file_name = f"{key}.pdf"
        self.zot.dump(key, file_name, "temp_folder")
        path = f"temp_folder/{file_name}"
        images = convert_from_path(
            path,
            thread_count=os.cpu_count() - 1,
            output_folder=Path(path).parents[0],
            paths_only=True,
            fmt="png"
        )
        metadata = { 'title': self.data['data']['title'], 'abstractNote': self.data['data']['abstractNote'], 'accessDate' : self.data['data']['dateAdded']}
        
        images = [Image.open(image_path) for image_path in images]
        metadata_images = [{**metadata, "page": i, "image_b64": Arxiv.get_base64(images[i])} for i in range(len(images))]
        return [(image, meta) for image, meta in zip(images,metadata_images)]  # TODO: process metadata for qdrant payload