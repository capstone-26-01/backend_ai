from django.contrib import admin

from api.models import AnalysisArtifact, AnalysisRun, Repository


@admin.register(Repository)
class RepositoryAdmin(admin.ModelAdmin):
    list_display = ('provider', 'full_name', 'default_branch', 'updated_at')
    search_fields = ('provider', 'owner', 'name', 'full_name')


@admin.register(AnalysisRun)
class AnalysisRunAdmin(admin.ModelAdmin):
    list_display = ('repository', 'revision', 'ref', 'status', 'started_at', 'finished_at', 'error_code')
    list_filter = ('status', 'repository__provider')
    search_fields = ('repository__full_name', 'revision', 'error_code')


@admin.register(AnalysisArtifact)
class AnalysisArtifactAdmin(admin.ModelAdmin):
    list_display = ('analysis_run', 'schema_version', 'node_count', 'edge_count', 'warning_count', 'created_at')
    search_fields = ('analysis_run__repository__full_name', 'analysis_run__revision', 'schema_version')
