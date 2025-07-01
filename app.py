from sentence_transformers import SentenceTransformer
from typing import List, Optional
from pydantic import BaseModel, Field
import inferless


@inferless.request
class RequestObjects(BaseModel):
    sentences: str = Field(default="Overview of climate change impacts on coastal cities")
    task: Optional[str] = "retrieval"
    prompt_name: Optional[str] = "query"

@inferless.response
class ResponseObjects(BaseModel):
    embeddings: List[float] = Field(default="Test output")

class InferlessPythonModel:
    def initialize(self):
        self.model = SentenceTransformer("jinaai/jina-embeddings-v4",trust_remote_code=True)

    def infer(self, inputs: RequestObjects) -> ResponseObjects:
        embs = self.model.encode(
            sentences=inputs.sentences,
            task=inputs.task,
            prompt_name=inputs.prompt_name,
        )                                            
        return ResponseObjects(embeddings=[v.tolist() for v in embs])

    def finalize(self):
      self.model = None
