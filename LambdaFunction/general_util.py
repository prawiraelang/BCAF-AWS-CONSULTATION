from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

def send_email(ses_client, sender, receiver, subject, body, file_content, file_name):
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ",".join(receiver)

    body_txt = MIMEText(body, "html")

    attachment = MIMEApplication(file_content)
    attachment.add_header("Content-Disposition", "attachment", filename=file_name)

    msg.attach(body_txt)
    msg.attach(attachment)

    response = ses_client.send_raw_email(
        Source=sender, Destinations=receiver, RawMessage={"Data": msg.as_string()}
    )

    return response