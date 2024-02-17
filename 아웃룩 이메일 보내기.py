import win32com.client
import email
import os

outlook = win32com.client.Dispatch("Outlook.Application")
mail = outlook.CreateItem(0)

# GetNamespace을 통해서 원하는 네이 공간 형태을 반환합니다.
# MAPI 만 지원합니다. 따라서 위의 영역은 그냥 고정된 영역이라고 보시면 됩니다.

mail.To = "etrezero@koreainvestment.com"
# email.CC = "etrezero@gmail.com"
mail.Subject = "매일매일 즐거운 하루가 되자"
mail.HTMLBody = """
<html>

<body lang=KO style='tab-interval:20.0pt;word-wrap:break-word'>
<div class=WordSection1>
<p class=MsoNormal>
안녕하십니까? 멀티에셋운용부 서재영 수석입니다. </p>
<p class=MsoNormal>
이 번 제안에 응해주셔서 감사합니다. 다음 번 제안도 잘 부탁드립니다.
</p>
</div>
</body>
</html>
"""

file = r'C:\Users\USER\Desktop\임시폴더\2jm57.xlsx'
mail.Attachments.Add(file)


# outlook.Display(True)

mail.send()

