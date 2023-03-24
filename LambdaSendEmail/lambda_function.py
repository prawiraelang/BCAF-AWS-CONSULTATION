import boto3
from logger_util import logger
from general_util import send_email

SUFFIX_DIR = '/with_header/'

def lambda_handler(event, context):
    try:
        s3_client = boto3.client("s3")
        ses_client = boto3.client("ses")
        
        for record in event["Records"]:
            # Prepare the data
            logger.info("Preparing the data...")
            event_time = record["eventTime"]
            event_name = record["eventName"]
            bucket = record["s3"]["bucket"]["name"]
            key = record["s3"]["object"]["key"]
            s3_path = f"s3://{bucket}/{key}"
            file_object = s3_client.get_object(Bucket=bucket, Key=key)
            file_content = file_object["Body"].read()
            file_name = key.split("/")[-1]
            
            if SUFFIX_DIR in s3_path:
                # Prepare the email
                logger.info("Preparing the email...")
                subject = f"Hasil Prediksi dari AWS"
                body ="<!DOCTYPE html> \
                <html> \
                <body> \
                <p> \
                Hi Team! \
                <br> \
                <br>" \
                f"This email is to notify you regarding <b>{event_name}</b> " \
                f"event at <b>{event_time}</b>. There is a new prediction stored " \
                f"on <b>{s3_path}</b>. Here I attach the file." \
                "<br> \
                <br> \
                Thanks, \
                <br> \
                AWS Lambda \
                </p> \
                </body> \
                </html>"
                
                message = {
                    "Subject": {"Data": subject}, 
                    "Body": {
                        "Html": {"Data": body}
                    }
                }
                sender = "komang.e.s.prawira@gdplabs.id"
                # receiver = [
                #     "komang.e.s.prawira@gdplabs.id",
                #     "rya.meyvriska@gdplabs.id"
                # ]
                receiver = [
                    "komang.e.s.prawira@gdplabs.id"
                ]
    
    
                # Send the email
                logger.info(f"Sending the email to {receiver}")
                response = send_email(
                    ses_client, sender, receiver, 
                    subject, body, file_content, file_name
                )
        
                if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
                    logger.info("Email sent successfully!")
            
    except Exception as e:
        logger.error(e)
        raise e
        
    return {
        "statusCode": 200,
        "message": f"{message}"
    }