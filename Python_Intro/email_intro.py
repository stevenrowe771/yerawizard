import yagmail

password = ""
with open("/home/steve/.local/share/.email_password", "r") as f:
    password = f.read().rstrip()

yag = yagmail.SMTP("steve771raspi@gmail.com", password)

yag.send(to="stevenrowe771@gmail.com",
         subject="Harry Potter Code",
         contents="Harry Potter Code",
         attachments="/home/steve/Desktop/New_Harry_Potter_Code/spell_model_nn.keras")

print("Email sent")