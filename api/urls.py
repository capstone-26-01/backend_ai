from django.urls import path
from . import views

#/api/repo/로 요청 오면 get_repo_file 함수 실행
urlpatterns = [
    path('analysis/', views.analysis),
    path('analysis/<int:analysis_id>/diff/', views.analysis_diff),
    path('analysis/<int:analysis_id>/', views.analysis_detail),
    path('diff/', views.graph_diff),
    path('share/', views.share),
    path('share/<str:share_id>/', views.share_detail),
    path('embed/<str:share_id>/', views.embed),
    path('repo/', views.get_repo_file),
    path('tree/', views.get_repo_tree),
    path('graph/', views.get_repo_graph),
    path('summary/', views.summary),
    path('node-summary/', views.node_summary),
    path('qa/', views.qa),
]
