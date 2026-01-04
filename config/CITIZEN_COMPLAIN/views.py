# Handle profile image upload
from django.views.decorators.csrf import csrf_exempt
@csrf_exempt
def upload_profile_image(request):
    if request.method == 'POST':
        user_id = request.session.get('user_id')
        if not user_id:
            return redirect('login')
        try:
            user = RegisteredUser.objects.get(id=user_id)
        except RegisteredUser.DoesNotExist:
            return redirect('login')
        if 'image' in request.FILES:
            user.image = request.FILES['image']
            user.save()
        return redirect('complaint')
    return redirect('complaint')
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
# My Complaints view

# Remove or restrict my_complaints_view so users can't access it directly
from django.http import HttpResponseForbidden

def my_complaints_view(request):
    # Option 1: Redirect to complaint page always
    return redirect('complaint')
from django.shortcuts import render, redirect
from django.utils import timezone
from .models import RegisteredUser, Complaint


# =========================
# HOME
# =========================
def home(request):
    return render(request, 'home.html')

# =========================
# ALL SERVICES
# =========================
def all_services(request):
    return render(request, 'all_services.html')

# =========================
# CONTACT
# =========================
def contact(request):
    return render(request, 'contact.html')


# =========================
# REGISTER
# =========================
def register(request):
    if request.method == 'POST':
        fullname = request.POST.get('fullname', '').strip()
        email = request.POST.get('email', '').strip()
        contact = request.POST.get('contact', '').strip()
        password = request.POST.get('password', '')
        confirm = request.POST.get('confirm_password', '')

        if not fullname or not email or not contact or not password:
            return render(request, 'register.html', {
                'error': 'Please fill in all required fields.'
            })

        if password != confirm:
            return render(request, 'register.html', {
                'error': 'Passwords do not match.'
            })

        if RegisteredUser.objects.filter(email=email).exists():
            return render(request, 'register.html', {
                'error': 'Email already registered.'
            })

        user = RegisteredUser.objects.create(
            fullname=fullname,
            email=email,
            contact=contact,
            password=password
        )

        request.session['user_id'] = user.id
        return redirect('complaint')

    return render(request, 'register.html')


# =========================
# LOGIN
# =========================
def login(request):
    if request.method == 'POST':
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '')

        if not email or not password:
            return render(request, 'login.html', {
                'error': 'Please enter email and password.'
            })

        try:
            user = RegisteredUser.objects.get(email=email, password=password)
        except RegisteredUser.DoesNotExist:
            return render(request, 'login.html', {
                'error': 'Invalid credentials.'
            })

        request.session['user_id'] = user.id
        return redirect('complaint')

    return render(request, 'login.html')


# =========================
# LOGOUT
# =========================
def logout(request):
    request.session.flush()
    return redirect('login')


# =========================
# COMPLAINT HOME (CATEGORY SELECTION)
# =========================
def complaint(request):
    user_id = request.session.get('user_id')
    if not user_id:
        return redirect('login')

    try:
        user = RegisteredUser.objects.get(id=user_id)
    except RegisteredUser.DoesNotExist:
        return redirect('login')

    # Handle complaint submission
    if request.method == 'POST':
        Complaint.objects.create(
            user=user,
            category=request.POST.get('category'),
            title=request.POST.get('title'),
            description=request.POST.get('description'),
            location=request.POST.get('location'),
            contact=request.POST.get('contact'),
            status='Pending',
            created_at=timezone.now()
        )
        return render(request, 'thank_you.html')

    # Show user's submitted complaints for status
    user_complaints = Complaint.objects.filter(user=user).order_by('-created_at')
    return render(request, 'complaint.html', {
        'registered_user': user,
        'user_complaints': user_complaints
    })


# =========================
# ECONOMY COMPLAINT
# =========================
def economy_view(request):
    user_id = request.session.get('user_id')
    if not user_id:
        return redirect('login')

    user = RegisteredUser.objects.get(id=user_id)


    if request.method == 'POST':
        Complaint.objects.create(
            user=user,
            category='Economy',
            title=request.POST.get('title'),
            description=request.POST.get('description'),
            location=request.POST.get('location'),
            contact=request.POST.get('contact'),
            status='Pending',
            created_at=timezone.now()
        )
        return render(request, 'thank_you.html')

    return render(request, 'economy.html', {
        'registered_user': user
    })


# =========================
# SOCIAL COMPLAINT
# =========================
def social_view(request):
    user_id = request.session.get('user_id')
    if not user_id:
        return redirect('login')

    user = RegisteredUser.objects.get(id=user_id)


    if request.method == 'POST':
        Complaint.objects.create(
            user=user,
            category='Social',
            title=request.POST.get('title'),
            description=request.POST.get('description'),
            location=request.POST.get('location'),
            contact=request.POST.get('contact'),
            status='Pending',
            created_at=timezone.now()
        )
        return render(request, 'thank_you.html')

    return render(request, 'social.html', {
        'registered_user': user
    })


# =========================
# POLITICS COMPLAINT
# =========================
def politics_view(request):
    user_id = request.session.get('user_id')
    if not user_id:
        return redirect('login')

    user = RegisteredUser.objects.get(id=user_id)


    if request.method == 'POST':
        Complaint.objects.create(
            user=user,
            category='Politics',
            title=request.POST.get('title'),
            description=request.POST.get('description'),
            location=request.POST.get('location'),
            contact=request.POST.get('contact'),
            status='Pending',
            created_at=timezone.now()
        )
        return render(request, 'thank_you.html')

    return render(request, 'politics.html', {
        'registered_user': user
    })


# =========================
# USER COMPLAINT STATUS & FEEDBACK
# =========================
def my_complaints(request):
    user_id = request.session.get('user_id')
    if not user_id:
        return redirect('login')

    user = RegisteredUser.objects.get(id=user_id)
    complaints = Complaint.objects.filter(user=user).order_by('-created_at')

    return render(request, 'my_complaints.html', {
        'registered_user': user,
        'complaints': complaints
    })
