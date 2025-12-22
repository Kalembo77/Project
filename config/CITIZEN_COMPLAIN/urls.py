from django.urls import path
from . import views  # <-- Make sure this line exists
from .views import upload_profile_image

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('login/', views.login, name='login'),
    path('complaint/', views.complaint, name='complaint'),

    # Category complaint pages
    path('economy/', views.economy_view, name='economy'),
    path('social/', views.social_view, name='social'),
    path('politics/', views.politics_view, name='politics'),
    # My Complaints page
    path('my-complaints/', views.my_complaints_view, name='my_complaints'),
    path('upload-profile-image/', upload_profile_image, name='upload_profile_image'),

    # All Services and Contact pages
    path('all-services/', views.all_services, name='all_services'),
    path('contact/', views.contact, name='contact'),
]
