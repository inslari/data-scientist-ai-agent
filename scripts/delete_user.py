import argparse
import sys
import os

# Add the parent directory (project root) to sys.path so 'api' module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.core.database import SessionLocal
from api.models.user import UserModel
def delete_user(email: str):
    db = SessionLocal()
    try:
        user = db.query(UserModel).filter(UserModel.email == email).first()
        if user:
            db.delete(user)
            db.commit()
            print(f"✅ User with email '{email}' has been successfully deleted.")
        else:
            print(f"❌ No user found with email '{email}'.")
    except Exception as e:
        db.rollback()
        print(f"⚠️ An error occurred: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete a user by email")
    parser.add_argument("email", type=str, help="The email address of the user to delete")
    args = parser.parse_args()
    
    delete_user(args.email)

# Example usage:
# uv run python scripts/delete_user.py "kullanici@mail.com"

