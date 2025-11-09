import base64
import logging
import mimetypes
import os
import os.path
import pickle
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient import errors
from googleapiclient.discovery import build
from googleapiclient.discovery_cache.base import Cache
from config import GMAIL_CLIENT_SECRET_PATH, GMAIL_TOKEN_PATH, EMAIL_RECIPIENT, STOCKS_CSV_PATH

class MemoryCache(Cache):
    _CACHE = {}

    def get(self, url):
        return MemoryCache._CACHE.get(url)

    def set(self, url, content):
        MemoryCache._CACHE[url] = content
        
def get_service():
    """Gets an authorized Gmail API service instance.

    Returns:
        An authorized Gmail API service instance..
    """    

    # If modifying these scopes, delete the file token.pickle.
    SCOPES = [
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.send',
    ]

    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(GMAIL_TOKEN_PATH):
        with open(GMAIL_TOKEN_PATH, 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                GMAIL_CLIENT_SECRET_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(GMAIL_TOKEN_PATH, 'wb') as token:
            pickle.dump(creds, token)

    service = build('gmail', 'v1', credentials=creds, cache=MemoryCache())
    return service

def my_tf_color_func(dictionary):
    def my_tf_color_func_inner(word, font_size, position, orientation, random_state=None, **kwargs):
        return "hsl(%d, 80%%, 55%%)" % (15 * dictionary[word])
    return my_tf_color_func_inner

def make_report( pred ):
    import pandas as pd 
    import matplotlib.pyplot as plt
    import tempfile
    from wordcloud import WordCloud
    
    stocks = [ ticker for prob, ticker in pred if prob > 0.98 ]
    df = pd.read_csv( STOCKS_CSV_PATH )

    sectors = dict(df[df['Ticker'].isin(stocks)]['Sector'].value_counts())
    
    # Create and generate a word cloud image:
    wc = WordCloud( background_color='black', color_func = my_tf_color_func( sectors ) ).generate_from_frequencies( sectors )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        wc.to_file(tmpfile.name)

    pred_table = ''.join(['<tr><td>{}</td><td>{}</td></tr>'.format(t,p) for p,t in pred])
    return '<table>{}</table>'.format(pred_table), tmpfile.name 
    
    
def send_message(service, sender, message):
  """Send an email message.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    message: Message to be sent.

  Returns:
    Sent Message.
  """
  try:
    sent_message = (service.users().messages().send(userId='me', body=message)
               .execute())
    logging.info('Message Id: %s', sent_message['id'])
    return sent_message
  except errors.HttpError as error:
    logging.error('An HTTP error occurred: %s', error)

def create_message(sender, to, subject, report):
    """Create a message for an email.

    Args:
    sender: Email address of the sender.
    to: Email address of the receiver.
    subject: The subject of the email message.
    message_text: The text of the email message.

    Returns:
    An object containing a base64url encoded email object.
    """
    msg = MIMEMultipart()
    msg["To"] = to
    msg["From"] = sender
    msg["Subject"] = subject

    body, attachment = report
    msgText = MIMEText('<p>%s</p><br><img src="cid:%s"><br>' % (body, attachment), 'html')  
    msg.attach(msgText)   # Added, and edited the previous line

    with open(attachment, 'rb') as fp:
        img = MIMEImage(fp.read())

    img.add_header('Content-ID', '<{}>'.format(attachment))
    msg.attach(img)

    s = msg.as_string()
    b = base64.urlsafe_b64encode(s.encode('utf-8'))
    return {'raw': b.decode('utf-8')}

def send( to=None, subject=None, body='hello' ):
    to = to or EMAIL_RECIPIENT
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=logging.INFO
    )

    try:
        if subject is None:
            from datetime import datetime
            subject = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        service = get_service()
        message = create_message("from@gmail.com", to, subject, make_report(body))
        send_message(service, "from@gmail.com", message)

    except Exception as e:
        logging.error(e)
        raise
