# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:12:28 2018

@author: TOM is in class
"""

from exchangelib import DELEGATE, Account, Credentials

credentials = Credentials(

    username='EXCHANGE\\sm_sewerci',  # Or myusername@example.com for O365

    password='EqyUQejBgMT32'
)
account = Account(primary_smtp_address='sewercameraimages@sew.com.au',credentials=credentials,

    autodiscover=True,

    access_type=DELEGATE
)

# Print first 100 inbox messages in reverse order

for m in messages:
    if (m.SenderEmailAddress == 'tomjellickal@gmail.com'): #&& m.Subject == "Loan account statement"):
        attachments = m.Attachments
        num_attach = len([x for x in attachments])
        for x in range(1, num_attach):
            attachment = attachments.Item(x)
            attachment.SaveASFile(os.path.join(get_path,attachment.FileName))
            print (attachment)
            message = messages.GetNext()

    else:
        message = messages.GetNext()
