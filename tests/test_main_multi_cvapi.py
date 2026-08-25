from fastapi.testclient import TestClient
from backend.main_multi import app

client = TestClient(app)

def test_main_app_has_cvapi_routes():
    res = client.post("/cvApi/open/api/cv/findModelPage", data={"size": 10})
    assert res.status_code == 200
    assert res.json()["code"] == 0
    assert res.json()["success"] is True

    res2 = client.post("/cvApi/open/api/cv/findCvWarningPage", data={"size": 5})
    assert res2.status_code == 200
    assert res2.json()["code"] == 0
    assert res2.json()["success"] is True

    res3 = client.post("/cvApi/open/api/cv/findCvWarningNumber", data={})
    assert res3.status_code == 200
    assert res3.json()["code"] == 0
    assert res3.json()["success"] is True
