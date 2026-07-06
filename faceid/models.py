from django.db import models

from .engine import ENGINE_CHOICES


class Person(models.Model):
    """One gallery entry = one enrolled identity for one backend.

    Mirrors the paper's one-shot enrollment (BAB II.2.2.3): a single
    embedding vector represents the person, so adding a new identity never
    requires retraining a model.
    """

    name = models.CharField(max_length=200)
    engine = models.CharField(
        max_length=20,
        choices=[(c, c) for c in ENGINE_CHOICES],
        help_text="Backend the embedding below was produced with.",
    )
    embedding = models.JSONField(help_text="L2-normalized embedding vector.")
    photo = models.ImageField(upload_to="gallery/", blank=True, null=True)
    det_score = models.FloatField(default=0.0)
    num_source_photos = models.PositiveSmallIntegerField(
        default=1,
        help_text="1 = pure one-shot, >1 if enrolled with TTA/multi-photo averaging.",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return f"{self.name} ({self.engine})"
