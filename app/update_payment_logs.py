from app import create_app, db
from app.models import PaymentLog
from stripe import Charge, PaymentIntent, BalanceTransaction

app = create_app()

with app.app_context():
    updated = 0
    for log in PaymentLog.query.all():
        if not log.stripe_payment_id:
            continue
        try:
            stripe_id = log.stripe_payment_id
            if stripe_id.startswith("pi_"):
                intent = PaymentIntent.retrieve(stripe_id)
                charge_id = intent.get("latest_charge")
            else:
                charge_id = stripe_id

            charge = Charge.retrieve(charge_id)
            balance_tx_id = charge.get("balance_transaction")
            balance_tx = BalanceTransaction.retrieve(balance_tx_id)

            log.amount = balance_tx.amount // 100
            log.fee = balance_tx.fee // 100
            log.net_income = balance_tx.net // 100
            updated += 1
        except Exception as e:
            print(f"❌ Error updating log {log.id}: {e}")

    db.session.commit()
    print(f"✅ Stripeベースで更新完了: {updated} 件")
