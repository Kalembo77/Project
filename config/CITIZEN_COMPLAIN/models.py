from django.db import models


class RegisteredUser(models.Model):
    fullname = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    contact = models.CharField(max_length=20)
    password = models.CharField(max_length=255)
    image = models.ImageField(upload_to='profile_images/', null=True, blank=True)

    def __str__(self):
        return self.fullname


class Complaint(models.Model):

    CATEGORY_CHOICES = [
        ('Economy', 'Economy'),
        ('Social', 'Social'),
        ('Politics', 'Politics'),
    ]

    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Accepted', 'Accepted'),
        ('In Progress', 'In Progress'),
        ('Resolved', 'Resolved'),
    ]

    user = models.ForeignKey(
        RegisteredUser,
        on_delete=models.CASCADE,
        related_name='complaints'
    )

    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES)

    title = models.CharField(max_length=200, null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    location = models.CharField(max_length=200, null=True, blank=True)
    contact = models.CharField(max_length=50, null=True, blank=True)

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='Pending'
    )

    admin_feedback = models.TextField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.fullname} - {self.category} - {self.status}"
