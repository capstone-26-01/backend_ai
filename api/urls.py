from django.urls import path
from . import views

#/api/repo/로 요청 오면 get_repo_file 함수 실행
urlpatterns = [
    path('repo/', views.get_repo_file),
    path('tree/', views.get_repo_tree),
    path('graph/', views.get_repo_graph),
    path('qa/', views.qa),
]