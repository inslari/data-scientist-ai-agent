import os
import resend

from src.core.logging_config import get_logger
logger = get_logger(__name__)

def send_verification_email(to_email: str, token: str, frontend_url: str = "http://localhost:5173") -> bool:
    """
    Sends an email verification link using Resend.
    Returns True if successful, False otherwise.
    """
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        logger.warning("Warning: RESEND_API_KEY is not set. Skipping verification email.")
        return False
        
    resend.api_key = api_key
    
    client_url = os.getenv("CLIENT_URL", "http://localhost:5173")
    verify_link = f"{client_url}/verify?token={token}"
    
    html_content = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #eaeaec; border-radius: 8px;">
        <h2 style="color: #333; margin-top: 0;">Verify your email address</h2>
        <p style="color: #555; line-height: 1.5;">
            Welcome to Inslari AI! To complete your registration and secure your account, please verify your email address by clicking the button below:
        </p>
        <div style="text-align: center; margin: 30px 0;">
            <a href="{verify_link}" style="background-color: #6366f1; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold; display: inline-block;">
                Verify Email
            </a>
        </div>
        <p style="color: #777; font-size: 13px; line-height: 1.5;">
            If the button doesn't work, you can copy and paste this link into your browser:<br>
            <a href="{verify_link}" style="color: #6366f1;">{verify_link}</a>
        </p>
        <hr style="border: 0; border-top: 1px solid #eaeaec; margin: 30px 0;">
        <p style="color: #999; font-size: 12px; margin-bottom: 0;">
            If you didn't create an account with Inslari AI, you can safely ignore this email.
        </p>
    </div>
    """
    
    try:
        response = resend.Emails.send({
            "from": "Inslari AI <info@inslari.com>",
            "to": to_email,
            "subject": "Verify your email address - Inslari AI",
            "html": html_content
        })
        logger.info(f"Verification email sent to {to_email}. Response: {response}")
        return True
    except Exception as e:
        logger.error(f"Failed to send verification email: {e}")
        return False

def send_reset_password_email(to_email: str, token: str) -> bool:
    """
    Sends a password reset email using Resend.
    """
    api_key = os.getenv("RESEND_API_KEY")
    logger.debug("RESEND_API_KEY set: %s", bool(api_key))
    if not api_key:
        logger.warning("Warning: RESEND_API_KEY is not set. Skipping reset password email.")
        return False
    logger.info("Sending reset password email to %s", to_email)
    resend.api_key = api_key

    client_url = os.getenv("CLIENT_URL", "http://localhost:5173")
    reset_link = f"{client_url}/reset-password?token={token}"
    
    html_content = f"""
    <div style="font-family: 'Inter', -apple-system, sans-serif; max-width: 480px; margin: 0 auto; padding: 40px 20px; text-align: center; background-color: #fcfcfd; border-radius: 16px; border: 1px solid #eaeaec;">
        <h1 style="color: #1a1a1c; font-size: 24px; font-weight: 700; margin-bottom: 12px; letter-spacing: -0.02em;">
            Reset your password
        </h1>
        <p style="color: #555; font-size: 15px; line-height: 1.6; margin-bottom: 32px;">
            We received a request to reset your Inslari AI password. If you didn't make this request, you can safely ignore this email.
        </p>
        <div style="margin-bottom: 32px;">
            <a href="{reset_link}" style="display: inline-block; background-color: #0f172a; color: #ffffff; font-size: 15px; font-weight: 600; text-decoration: none; padding: 14px 32px; border-radius: 8px; transition: background-color 0.2s;">
                Reset Password
            </a>
        </div>
        <p style="color: #777; font-size: 13px; line-height: 1.5;">
            If the button doesn't work, you can copy and paste this link into your browser:<br>
            <a href="{reset_link}" style="color: #6366f1;">{reset_link}</a>
        </p>
        <hr style="border: 0; border-top: 1px solid #eaeaec; margin: 30px 0;">
        <p style="color: #999; font-size: 12px; margin-bottom: 0;">
            This link will expire in 1 hour.
        </p>
    </div>
    """
    
    try:
        response = resend.Emails.send({
            "from": "Inslari AI <info@inslari.com>",
            "to": to_email,
            "subject": "Reset your password - Inslari AI",
            "html": html_content
        })
        logger.info(f"Reset password email sent to {to_email}. Response: {response}")
        return True
    except Exception as e:
        logger.error(f"Failed to send reset password email: {e}")
        return False
