from django.contrib import admin
from .models import RegisteredUser, Complaint


@admin.register(RegisteredUser)
class RegisteredUserAdmin(admin.ModelAdmin):
    list_display = ('fullname', 'email', 'contact')
    search_fields = ('fullname', 'email')


@admin.register(Complaint)
class ComplaintAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'category', 'status', 'created_at')
    list_filter = ('status', 'category')
    search_fields = ('title', 'description', 'user__fullname')
    list_editable = ('status',)
