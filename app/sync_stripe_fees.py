import stripe
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL)

with engine.connect() as conn:
    result = conn.execute(text("SELECT id, stripe_payment_id FROM payment_log"))
    payments = result.fetchall()

updates = []
for row in payments:
    local_id, stripe_pid = row
    try:
        intent = stripe.PaymentIntent.retrieve(stripe_pid)
        charge_id = intent.get("latest_charge")
        if not charge_id:
            continue
        charge = stripe.Charge.retrieve(charge_id)
        balance_tx = stripe.BalanceTransaction.retrieve(charge["balance_transaction"])
        updates.append((balance_tx.fee, balance_tx.net, local_id))
    except Exception as e:
        print(f"❌ {stripe_pid}: {e}")

with engine.begin() as conn:
    for fee, net, local_id in updates:
        conn.execute(
            text("UPDATE payment_log SET fee = :fee, net_income = :net WHERE id = :id"),
            {"fee": fee, "net": net, "id": local_id}
        )

print(f"✅ 更新完了：{len(updates)} 件")
