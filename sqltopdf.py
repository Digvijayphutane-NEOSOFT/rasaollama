import pymysql
from reportlab.pdfgen import canvas

# Connect to the 'bank' database
conn = pymysql.connect(
    host="localhost",
    database="bank",
    user="root",
    password="password",
    port=3306
)

# Retrieve the data from the 'bank' database
cur = conn.cursor()
cur.execute("SELECT Transaction_Number, Date, Amount_change, Balance_after, Receiver, Information FROM Transactions")
rows = cur.fetchall()

# Create the report
pdf = canvas.Canvas("bank_transaction_report.pdf")
pdf.setFont("Helvetica-Bold", 16)
pdf.drawString(50, 800, "Bank Transaction Report")
pdf.setFont("Helvetica", 12)
y = 750

for row in rows:
    pdf.drawString(50, y, "Transaction Number: " + str(row[0]))
    pdf.drawString(50, y - 20, "Date: " + str(row[1]))
    pdf.drawString(50, y - 40, "Amount Change: " + str(row[2]))
    pdf.drawString(50, y - 60, "Balance After: " + str(row[3]))
    pdf.drawString(50, y - 80, "Receiver: " + row[4])
    pdf.drawString(50, y - 100, "Information: " + row[5])
    y -= 120
    if y < 50:  # Check if the y-coordinate is too low for the next row
        pdf.showPage()  # Create a new page
        pdf.setFont("Helvetica", 12)
        y = 800

pdf.save()

# Close the database connection
cur.close()
conn.close()
