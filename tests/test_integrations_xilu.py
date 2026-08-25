from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.integrations.router import router as integrations_router
from backend.integrations.xilu.service import XiluApiService

app = FastAPI()
app.include_router(integrations_router)
client = TestClient(app)

def test_xilu_service():
    model_data = XiluApiService.get_model_page()
    assert "records" in model_data
    assert len(model_data["records"]) > 0

    number_data = XiluApiService.get_warning_number()
    assert "todayWarningNumber" in number_data

def test_xilu_endpoints_via_integrations_router():
    res = client.post("/cvApi/open/api/cv/findModelPage", data={"size": 10})
    assert res.status_code == 200
    assert res.json()["code"] == 0

    res2 = client.post("/cvApi/open/api/cv/findCvWarningPage", data={"size": 5})
    assert res2.status_code == 200
    assert res2.json()["code"] == 0

    res3 = client.post("/cvApi/open/api/cv/findCvWarningNumber", data={})
    assert res3.status_code == 200
    assert res3.json()["code"] == 0
