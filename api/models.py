from django.db import models


class Repository(models.Model):
    provider = models.CharField(max_length=32, default='github')
    owner = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    full_name = models.CharField(max_length=511)
    default_branch = models.CharField(max_length=255, blank=True, null=True)
    clone_url = models.URLField(max_length=1024, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['provider', 'full_name'], name='unique_repository_provider_full_name'),
        ]
        ordering = ['provider', 'full_name']

    def __str__(self):
        return f'{self.provider}:{self.full_name}'


class AnalysisRun(models.Model):
    STATUS_STARTED = 'started'
    STATUS_SUCCEEDED = 'succeeded'
    STATUS_FAILED = 'failed'
    STATUS_CHOICES = [
        (STATUS_STARTED, 'Started'),
        (STATUS_SUCCEEDED, 'Succeeded'),
        (STATUS_FAILED, 'Failed'),
    ]

    repository = models.ForeignKey(Repository, on_delete=models.CASCADE, related_name='analysis_runs')
    ref = models.CharField(max_length=255, default='HEAD')
    revision = models.CharField(max_length=255, blank=True)
    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default=STATUS_STARTED)
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(blank=True, null=True)
    error_code = models.CharField(max_length=64, blank=True)
    error_message = models.TextField(blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=['repository', 'revision'],
                condition=models.Q(status='succeeded'),
                name='unique_succeeded_analysis_run_per_revision',
            ),
        ]
        indexes = [
            models.Index(fields=['repository', 'revision', 'status']),
        ]
        ordering = ['-started_at']

    def __str__(self):
        return f'{self.repository.full_name}@{self.revision or "unknown"} {self.status}'


class AnalysisArtifact(models.Model):
    analysis_run = models.OneToOneField(AnalysisRun, on_delete=models.CASCADE, related_name='artifact')
    schema_version = models.CharField(max_length=64)
    payload = models.JSONField()
    node_count = models.PositiveIntegerField(default=0)
    edge_count = models.PositiveIntegerField(default=0)
    warning_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.analysis_run.repository.full_name}@{self.analysis_run.revision}'
