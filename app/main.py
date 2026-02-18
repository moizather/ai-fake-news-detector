from fastapi import FastAPI, Request, Response
from app.model import predict_text
from twilio.twiml.messaging_response import MessagingResponse

app = FastAPI()

@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    form_data = await request.form()
    incoming_msg = form_data.get("Body", "")

    prediction = predict_text(incoming_msg)

    twilio_response = MessagingResponse()
    twilio_response.message(prediction)

    return Response(
        content=str(twilio_response),
        media_type="application/xml"
    )
