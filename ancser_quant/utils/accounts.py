import os
from dotenv import load_dotenv
from typing import List

def get_configured_accounts() -> List[str]:
    """Read .env or OS environment variables and return a list of available account names."""
    
    # Force reload of .env to pick up real-time manual updates while Streamlit is alive
    load_dotenv(override=True)
    
    accounts = []
    
    # 1. Check Default "Main" Account
    if os.getenv("APCA_API_KEY_ID") and os.getenv("APCA_API_SECRET_KEY"):
        if "Main" not in accounts:
            accounts.append("Main")
            
    # 2. Scan for other suffix accounts (e.g. APCA_API_KEY_ID_TEST, APCA_API_KEY_ID_SECOND)
    for key in os.environ.keys():
        if key.startswith("APCA_API_KEY_ID_"):
            suffix = key.replace("APCA_API_KEY_ID_", "")
            
            # If the matching secret exists (handle both _SECRET_KEY_ and _SECRET_ formatting)
            secret_key_var1 = f"APCA_API_SECRET_KEY_{suffix}"
            secret_key_var2 = f"APCA_API_SECRET_{suffix}"
            
            if os.getenv(secret_key_var1) or os.getenv(secret_key_var2):
                # We can formalize the account name. E.g. MWU4Factor -> mwu4factor -> MWU4Factor
                acc_name = suffix
                # Default APCA_API_KEY_ID_MAIN could clash with "Main", avoid duplicates
                if acc_name.lower() == "main":
                    acc_name = "Main"
                    
                if acc_name not in accounts:
                    accounts.append(acc_name)
                    
    # Fallback to prevent UI crash if no keys are found
    if not accounts:
        accounts.append("Main")
        
    return accounts
