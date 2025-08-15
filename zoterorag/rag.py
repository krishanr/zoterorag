from unsloth import FastVisionModel
import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers import TextStreamer

from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Filter, FieldCondition, MatchValue

import time
from tqdm import tqdm
import stamina
import os


from .datamodel import Arxiv
from dotenv import load_dotenv

load_dotenv()

class RAG():

    def __init__(self, qdrant_url, collection_name, retrieval_model, device="cuda:0"):
        if retrieval_model != "nomic-ai/nomic-embed-multimodal-3b":
            raise Exception(f"RAG only works with model nomic-ai/nomic-embed-multimodal-3b")
        
        self.qdrant_client = QdrantClient(qdrant_url)
        self.collection_name = collection_name
        self.device = device

        self.colpali_model = BiQwen2_5.from_pretrained(
            retrieval_model,
            torch_dtype=torch.bfloat16,
            device_map=self.device,  # or "mps" if on Apple Silicon
            attn_implementation= None ,#"flash_attention_2" if is_flash_attn_2_available() else None,
            token=os.getenv("HUGGINGFACE_ACCESS_TOKEN")
        ).eval()

        self.colpali_processor = BiQwen2_5_Processor.from_pretrained(retrieval_model,device_map=self.device,token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"))

        self.model, self.tokenizer = None, None

    def create_collection(self):
        #TODO: check if collection exists
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            on_disk_payload=False,  # store the payload on disk
            vectors_config=models.VectorParams(
                size=2048,
                distance=models.Distance.COSINE,
                on_disk=False, 
                quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=True  # keep only quantized vectors in RAM
                    ),
                ),
            ),
        )


    @stamina.retry(on=Exception, attempts=3) # retry mechanism if an exception occurs during the operation
    def upsert_to_qdrant(self,points):
        self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=False,
                )
        return True
    
    def index_dataset(self,dataset,type : str = "text",batch_size = 4):
        total = len(dataset)
        with tqdm(total=total, desc="Indexing Progress") as pbar:
            for i in range(0, total, batch_size):
                batch = dataset[i : i + batch_size]

                # Upload points to Qdrant
                try:
                    # The images are already PIL Image objects, so we can use them directly
                    data = [ item[0] for item in  batch]
                    
                except Exception as e:
                    print(f"Error collecting query at index {i}: {e}")
                    continue

                # Process and encode images
                with torch.no_grad():
                    if type == "text":
                        max_len = 30000
                        batch_queries = self.colpali_processor.process_queries([ item[:max_len] for item in  data]).to(self.colpali_model.device)
                        embeddings = self.colpali_model(**batch_queries)
                    else: # type == "image"
                        batch_images = self.colpali_processor.process_images(data).to(
                            self.colpali_model.device
                        )
                        embeddings = self.colpali_model(**batch_images)

                multivectors = embeddings.cpu().float().numpy().tolist()
                del embeddings
                # Prepare points for Qdrant
                points = []
                for j, (multivector, metadata) in enumerate(zip(multivectors,[ item[1] for item in  batch])):
                    # Convert the embedding to a list of vectors
                    #multivector = embedding.cpu().float().numpy().tolist()
                    points.append(
                        models.PointStruct(
                            id=i + j,  # we just use the index as the ID
                            vector=multivector,  # This is now a list of vectors
                            payload=metadata,  # can also add other metadata/data
                        )
                    )

                # Upload points to Qdrant
                try:
                    self.upsert_to_qdrant(points)
                except Exception as e:
                    print(f"Error during upsert: {e}")
                    continue

                # Update the progress bar
                pbar.update(batch_size)

        print("Indexing complete!")
        self.qdrant_client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=10),
        )

    def search(self,query_text, top_k = 10):
        with torch.no_grad():
            
            batch_query = self.colpali_processor.process_queries([query_text]).to(
                self.colpali_model.device
            )
            query_embedding = self.colpali_model(**batch_query)

        multivector_query = query_embedding[0].cpu().float().numpy().tolist()

        # Query qdrant database
        start_time = time.time()
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=multivector_query,
            with_payload=True,
            limit=top_k,
            timeout=100,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=2.0,
                )
            )
        )
        end_time = time.time()
        # Search in Qdrant
        search_result.points

        elapsed_time = end_time - start_time
        print(f"Qdrant search completed in {elapsed_time:.4f} seconds")

        return search_result
    

    def get_images(self, search_result):
        return [(Arxiv.get_img(result.payload.get('image_b64')),result.payload.get("title")[:30] + f"... | Page {result.payload.get('page')+1}")  for result in search_result.points if result.payload.get('image_b64')]

    def generate(self,query_text, search_result, top_k_text = 2, top_k_images = 2, text_cutoff=20000,device : str ="cuda"):
        text_query = """Here is the text query:

        {query}

        """
        i = 1
        text_query = text_query.replace("{query}", query_text)
        for result in search_result.points:
            if i > top_k_text:
                break
            if result.payload.get('fulltext'):
                text_query += f"Here is text context {i}: \n" + result.payload.get('fulltext')[:text_cutoff] + "\n"
            i += 1

        print(f"Text query has length {len(text_query)}.")

        if self.model is None:
            self.model, self.tokenizer = FastVisionModel.from_pretrained("unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
                                            device_map=device,
                                            load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
                                            use_gradient_checkpointing = "unsloth",token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"))
            
        messages = [
        {"role": "system", "content": "You are an expert retrieval augmented generation agent, capable of creating clear answers using text and images."},
        {
            "role": "user",
            "content": 
        [{"type": "text", "text": text_query},
         *[{"type": "image"  } for result in search_result.points if result.payload.get('image_b64')][:top_k_images],
        ]}]
        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt = True)

        FastVisionModel.for_inference(self.model) # Enable for inference!

        images = [Arxiv.get_img(result.payload.get('image_b64')) for result in search_result.points if result.payload.get('image_b64')][:top_k_images]
        if images:
            inputs = self.tokenizer(
                images,
                input_text,
                add_special_tokens = False,
                return_tensors = "pt",
            ).to(device)
        else:
            inputs =  self.tokenizer(
                None,
                input_text,
                add_special_tokens = False,
                return_tensors = "pt",
            ).to(device)


        #text_streamer = TextStreamer( self.tokenizer, skip_prompt = True)
        generated_ids =  self.model.generate(**inputs, max_new_tokens = 512,
                        use_cache = False, temperature = 1.5, min_p = 0.1)
        generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]):]
        return self.processor.decode(generated_ids_trimmed, skip_special_tokens=True), text_query