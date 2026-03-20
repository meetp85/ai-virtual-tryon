import random
import string
from datetime import datetime, timedelta

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session, flash
from flask_login import login_user, logout_user, login_required, current_user

from models import db, User, OTPRecord
import config

auth_bp = Blueprint('auth', __name__)


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def generate_otp():
    """Generate a random 6-digit OTP."""
    return ''.join(random.choices(string.digits, k=config.OTP_LENGTH))


def send_sms_otp(phone, otp_code):
    """
    Send OTP via Twilio SMS.
    Returns True if sent, False if failed.
    """
    try:
        from twilio.rest import Client
        client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f"Your Parshva Jewellers verification code is: {otp_code}. Valid for 5 minutes.",
            from_=config.TWILIO_PHONE_NUMBER,
            to=phone
        )
        print(f"[SMS] OTP sent to {phone}: SID {message.sid}")
        return True
    except Exception as e:
        print(f"[SMS ERROR] {e}")
        # In development, print OTP to console so you can still test
        print(f"[DEV] OTP for {phone}: {otp_code}")
        return False


def create_and_send_otp(phone, purpose):
    """
    Create OTP record in DB and send via SMS.
    Returns (success: bool, message: str, otp_for_dev: str)
    """
    otp_code = generate_otp()
    expires_at = datetime.utcnow() + timedelta(seconds=config.OTP_EXPIRY_SECONDS)

    # Invalidate any previous unused OTPs for this phone
    OTPRecord.query.filter_by(phone=phone, is_used=False).update({'is_used': True})

    # Save new OTP
    otp_record = OTPRecord(
        phone=phone,
        otp_code=otp_code,
        purpose=purpose,
        expires_at=expires_at
    )
    db.session.add(otp_record)
    db.session.commit()

    # Send SMS
    sms_sent = send_sms_otp(phone, otp_code)

    if sms_sent:
        return True, "OTP sent to your phone.", otp_code
    else:
        # Still return success — OTP is in DB, user can use dev console output
        return True, "OTP generated (check console if SMS not configured).", otp_code


def verify_otp(phone, otp_code, purpose):
    """
    Verify OTP from database.
    Returns (success: bool, message: str)
    """
    otp_record = OTPRecord.query.filter_by(
        phone=phone,
        otp_code=otp_code,
        purpose=purpose,
        is_used=False
    ).order_by(OTPRecord.created_at.desc()).first()

    if not otp_record:
        return False, "Invalid OTP. Please try again."

    if datetime.utcnow() > otp_record.expires_at:
        otp_record.is_used = True
        db.session.commit()
        return False, "OTP has expired. Please request a new one."

    # Mark as used
    otp_record.is_used = True
    db.session.commit()
    return True, "OTP verified successfully."


# -------------------------------------------------------------------
# ROUTES — PAGES
# -------------------------------------------------------------------

@auth_bp.route('/auth')
def auth_page():
    """Render the sign in / sign up page."""
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return render_template('auth.html')


@auth_bp.route('/profile')
@login_required
def profile_page():
    """Simple profile page."""
    return render_template('profile.html')


@auth_bp.route('/logout')
def logout():
    """Log out and redirect home."""
    logout_user()
    session.clear()
    return redirect(url_for('home'))


# -------------------------------------------------------------------
# API ROUTES — SIGN UP
# -------------------------------------------------------------------

@auth_bp.route('/api/auth/signup', methods=['POST'])
def api_signup():
    """Step 1: Register user + send OTP."""
    data = request.json
    full_name = data.get('full_name', '').strip()
    phone = data.get('phone', '').strip()
    email = (data.get('email') or '').strip() or None

    # Validation
    if not full_name or len(full_name) < 2:
        return jsonify({'success': False, 'message': 'Please enter your full name.'})

    if not phone or len(phone) < 10:
        return jsonify({'success': False, 'message': 'Please enter a valid phone number.'})

    # Ensure phone has country code
    if not phone.startswith('+'):
        phone = '+91' + phone  # Default to India

    # Check if user already exists
    existing = User.query.filter_by(phone=phone).first()
    if existing and existing.is_verified:
        return jsonify({'success': False, 'message': 'This phone number is already registered. Please sign in.'})

    # Create or update user
    if existing:
        existing.full_name = full_name
        existing.email = email
    else:
        new_user = User(full_name=full_name, phone=phone, email=email)
        db.session.add(new_user)

    db.session.commit()

    # Send OTP
    success, message, otp = create_and_send_otp(phone, 'signup')

    return jsonify({
        'success': success,
        'message': message,
        'dev_otp': otp  # Remove this in production!
    })


# -------------------------------------------------------------------
# API ROUTES — VERIFY OTP (both signup and login)
# -------------------------------------------------------------------

@auth_bp.route('/api/auth/verify-otp', methods=['POST'])
def api_verify_otp():
    """Step 2: Verify OTP and log user in."""
    data = request.json
    phone = data.get('phone', '').strip()
    otp_code = data.get('otp', '').strip()
    purpose = data.get('purpose', 'login')  # 'signup' or 'login'

    if not phone.startswith('+'):
        phone = '+91' + phone

    # Verify OTP
    success, message = verify_otp(phone, otp_code, purpose)

    if not success:
        return jsonify({'success': False, 'message': message})

    # Find user
    user = User.query.filter_by(phone=phone).first()
    if not user:
        return jsonify({'success': False, 'message': 'User not found. Please sign up first.'})

    # Mark as verified + update last login
    user.is_verified = True
    user.last_login = datetime.utcnow()
    db.session.commit()

    # Log in with Flask-Login
    login_user(user, remember=True)

    return jsonify({
        'success': True,
        'message': 'Welcome, ' + user.full_name + '!',
        'user': {
            'name': user.full_name,
            'phone': user.phone
        }
    })


# -------------------------------------------------------------------
# API ROUTES — SIGN IN (existing user)
# -------------------------------------------------------------------

@auth_bp.route('/api/auth/login', methods=['POST'])
def api_login():
    """Send OTP to existing user for login."""
    data = request.json
    phone = data.get('phone', '').strip()

    if not phone or len(phone) < 10:
        return jsonify({'success': False, 'message': 'Please enter a valid phone number.'})

    if not phone.startswith('+'):
        phone = '+91' + phone

    # Check user exists
    user = User.query.filter_by(phone=phone, is_verified=True).first()
    if not user:
        return jsonify({'success': False, 'message': 'No account found with this number. Please sign up.'})

    # Send OTP
    success, message, otp = create_and_send_otp(phone, 'login')

    return jsonify({
        'success': success,
        'message': message,
        'user_name': user.full_name,
        'dev_otp': otp  # Remove in production!
    })


# -------------------------------------------------------------------
# API ROUTES — RESEND OTP
# -------------------------------------------------------------------

@auth_bp.route('/api/auth/resend-otp', methods=['POST'])
def api_resend_otp():
    """Resend OTP to phone."""
    data = request.json
    phone = data.get('phone', '').strip()
    purpose = data.get('purpose', 'login')

    if not phone.startswith('+'):
        phone = '+91' + phone

    success, message, otp = create_and_send_otp(phone, purpose)

    return jsonify({
        'success': success,
        'message': 'New OTP sent.',
        'dev_otp': otp
    })


# -------------------------------------------------------------------
# API — CHECK AUTH STATUS
# -------------------------------------------------------------------

@auth_bp.route('/api/auth/status')
def api_auth_status():
    """Check if user is logged in (for JS)."""
    if current_user.is_authenticated:
        return jsonify({
            'logged_in': True,
            'user': {
                'name': current_user.full_name,
                'phone': current_user.phone
            }
        })
    return jsonify({'logged_in': False})