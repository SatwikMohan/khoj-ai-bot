from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="Conversation role, usually user or assistant.")
    content: str = Field(..., min_length=1, description="Message text.")


class QARequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question for the QA bot.")
    top_k: int = Field(8, ge=1, le=10, description="Number of context chunks to retrieve.")
    temperature: float = Field(0.55, ge=0, le=1, description="Controls response warmth/variation.")
    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Recent conversation turns used only to interpret follow-up questions.",
    )


class SourceChunk(BaseModel):
    source: str | None = None
    folder_path: str | None = None
    page: int | None = None
    chunk_index: int | None = None
    relevance_score: float | None = None
    preview: str


class QAResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    query_type: str = "document"


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text or Markdown to synthesize.")
    voice_id: str | None = Field(None, description="Optional local voice id/name, or Edge voice name when TTS_ENGINE=edge.")
    ref_audio: str | None = Field(None, description="Reserved for API compatibility; local TTS does not use reference audio.")
    model: str | None = Field(None, description="Reserved for API compatibility; TTS selects the configured local or Edge engine.")
    tone: str = Field("neutral", description="Tone preset: neutral, warm, cheerful, calm, serious, energetic, or custom.")
    rate: str = Field("+0%", description="Speech rate adjustment, for example +10% or -10%.")
    pitch: str = Field("+0Hz", description="Speech pitch adjustment, for example +2Hz or -2Hz.")
    volume: str = Field("+0%", description="Speech volume adjustment, for example +10% or -10%.")
    response_format: str = Field("wav", description="Requested audio format. Local TTS returns WAV; Edge TTS returns MP3.")
    max_words: int = Field(260, ge=40, le=900, description="Maximum words to send to TTS.")
