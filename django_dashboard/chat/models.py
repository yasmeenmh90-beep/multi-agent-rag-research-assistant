from django.db import models


class Conversation(models.Model):
    """One conversation thread. session_id links back to the FastAPI
    backend's in-memory chat history store (see app/main.py SESSIONS),
    so the agent pipeline's contextualizer can resolve follow-up questions."""
    session_id = models.CharField(max_length=64, blank=True, default="")
    title = models.CharField(max_length=200, blank=True, default="New conversation")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return self.title or f"Conversation {self.pk}"


class Message(models.Model):
    """One question/answer turn within a conversation, with the metadata
    the agent pipeline returned (sources, domains, grounded status)."""
    conversation = models.ForeignKey(Conversation, related_name="messages", on_delete=models.CASCADE)
    question = models.TextField()
    answer = models.TextField(blank=True, default="")
    sub_questions = models.JSONField(default=list, blank=True)
    domains_used = models.JSONField(default=list, blank=True)
    sources = models.JSONField(default=list, blank=True)
    is_grounded = models.BooleanField(default=False)
    retried = models.BooleanField(default=False)  # True if the critic rejected
    response_time_s = models.FloatField(default=0)
    # the first draft and the rewriter reformulated + re-retrieved before
    # this final answer was produced (detected via an SSE 'restart' event).
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return self.question[:60]


class EvalRun(models.Model):
    """One RAGAS evaluation run (app/eval/ragas_eval.py), triggered from
    the dashboard and stored so scores are trackable over time instead of
    only visible in a terminal or CSV."""
    created_at = models.DateTimeField(auto_now_add=True)
    num_questions = models.IntegerField()
    faithfulness_avg = models.FloatField()
    context_precision_avg = models.FloatField()
    per_question = models.JSONField(default=list, blank=True)
 
    class Meta:
        ordering = ["-created_at"]
 
    def __str__(self):
        return f"Eval {self.created_at:%Y-%m-%d %H:%M} - faithfulness {self.faithfulness_avg}"
 