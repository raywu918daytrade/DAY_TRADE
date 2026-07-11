"""富邦連線測試：確認身分證字號＋密碼＋憑證可正常登入。"""
try:
    from fubon.trade_api import login
except ImportError:
    from trade_api import login

if __name__ == "__main__":
    sdk, accounts = login()
    print("登入成功，帳戶：")
    for acc in accounts:
        print(f"  {acc.name}  {acc.branch_no}-{acc.account}  ({acc.account_type})")
