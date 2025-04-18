import requests
import logging


class APIClient:
    def __init__(self, base_url='http://gatehub.runasp.net'):
        self.base_url = base_url.rstrip('/')

    def send_fine_data(self, plate_number, gate_id, fine_value=0, fine_type="No Fine"):
        """Send fine data to API"""
        data = {
            "plateNumber": plate_number,
            "fineValue": fine_value,
            "fineType": fine_type,
            "gateId": gate_id
        }


        try:
            endpoint = f"{self.base_url}/api/GateStaff/AddFine"
            response = requests.post(
                endpoint,
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )

            if response.status_code == 200:
                print(f"✅ API Response: {response.json()}")
                return response.json()
            else:
                print(f"❌ API Error: Status {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ API Error: {str(e)}")
            return None

    # Alias for compatibility
    def send_data(self, data):
        """Alternative method for sending data directly"""
        try:
            endpoint = f"{self.base_url}/api/GateStaff/AddFine"
            response = requests.post(
                endpoint,
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )

            if response.status_code == 200:
                print(f"✅ API Response: {response.json()}")
                return response.json()
            else:
                print(f"❌ API Error: Status {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ API Error: {str(e)}")
            return None