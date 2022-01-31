import urllib, requests
from datetime import timedelta
from functools import wraps
import os, traceback, time
import socket


def send_message(text, chat_id, file=None, filename="", reply_markup=None, parse_markdown=False):
    text = text.replace("_", "-")
    text = urllib.parse.quote_plus(text).replace("*", "%2A")
    url = f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage?text={text}&chat_id={chat_id}"
    if reply_markup: url += f"&reply_markup={reply_markup}"
    if parse_markdown: url += f"&parse_mode=Markdown"
    kwargs = {'files': {'document': (filename, file)}} if filename and file else {}
    response = requests.get(url, **kwargs)
    assert response.status_code == 200
    return response.content.decode("UTF-8")


def telegram_notify(only_terminal=True, only_on_fail=True, log_start=False):
    """only_terminal means only send telegram-messages on fail for non-interactive sessions. Everything else messes with the debugger"""
    catch_exceptions = not ("PYCHARM_HOSTED" in os.environ) if only_terminal else True
    def actual_decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                ts = time.time()
                if log_start:
                    send_message(f"Function {fn.__name__} started on {socket.gethostname()}!",  os.environ["TELEGRAM_MY_CHAT_ID"])
                res = fn(*args, **kwargs)
            except Exception as e:
                te = time.time()
                send_message(f"Function {fn.__name__} on {socket.gethostname()} failed after {timedelta(seconds=round(te-ts))}",  os.environ["TELEGRAM_MY_CHAT_ID"])
                error_args = "\n".join(e.args)
                send_message(f"Exception: {e.__repr__()} \n Args: {error_args}",  os.environ["TELEGRAM_MY_CHAT_ID"])
                send_message(f"Traceback: \n {traceback.format_exc()}",  os.environ["TELEGRAM_MY_CHAT_ID"])
                raise e
            else:
                if not only_on_fail:
                    te = time.time()
                    send_message(f"Function {fn.__name__} on {socket.gethostname()} is done after {timedelta(seconds=round(te-ts))}",  os.environ["TELEGRAM_MY_CHAT_ID"])
                return res
        return wrapped if catch_exceptions else fn
    return actual_decorator


if __name__ == "__main__":
    MY_CHAT_ID = os.environ["TELEGRAM_MY_CHAT_ID"]
    TGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('message')
    args = parser.parse_args()
    send_message(args.message, MY_CHAT_ID)