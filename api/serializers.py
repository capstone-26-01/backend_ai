from rest_framework import serializers


def extract_repo_path(repo_url):
    parts = repo_url.rstrip('/').split('github.com/')
    if len(parts) < 2:
        return None
    return parts[1]


class RepoUrlSerializer(serializers.Serializer):
    repo_url = serializers.CharField()

    def validate_repo_url(self, value):
        repo_path = extract_repo_path(value)
        if not repo_path:
            raise serializers.ValidationError('올바른 GitHub URL이 아닙니다')
        return repo_path  # 검증 후 repo_path로 변환해서 반환


class QASerializer(serializers.Serializer):
    repo_url = serializers.CharField()
    question = serializers.CharField()

    def validate_repo_url(self, value):
        repo_path = extract_repo_path(value)
        if not repo_path:
            raise serializers.ValidationError('올바른 GitHub URL이 아닙니다')
        return repo_path