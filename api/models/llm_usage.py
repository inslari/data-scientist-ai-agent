import time

from sqlalchemy import Column, Integer, String

from api.core.database import Base


class LLMUsageModel(Base):
    """Per-run LLM token usage, aggregated by (project, provider, model) (C3).

    The `UsageTrackingCallback` (src/core/llm_usage.py) captures `usage_metadata`
    from every LLM call during a worker job and flushes one aggregated row per
    (project_id, provider, model) at job end. This is the data the roadmap
    'cost budget' feature and any per-project/user spend reporting build on.

    Tokens are summed; `calls` counts how many LLM invocations rolled up here.
    """

    __tablename__ = "llm_usage"

    id = Column(Integer, primary_key=True, index=True)

    # Epoch seconds (matches BackgroundJobModel / LogEntryModel convention).
    created_at = Column(Integer, default=lambda: int(time.time()), index=True)

    project_id = Column(String, nullable=True, index=True)
    provider = Column(String, nullable=True, index=True)
    model = Column(String, nullable=True, index=True)

    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    calls = Column(Integer, default=0)
