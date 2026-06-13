"""
Disposable / temporary email domain blocklist.

Sources merged from multiple public blocklists.
Add new domains as needed.
"""

DISPOSABLE_EMAIL_DOMAINS: set[str] = {
    # --- mailnull / classic ---
    "mailnull.com", "trashmail.com", "trashmail.net", "trashmail.io",
    "trashmail.me", "trashmail.at", "trashmail.org",
    "guerrillamail.com", "guerrillamail.net", "guerrillamail.org",
    "guerrillamail.biz", "guerrillamail.de", "guerrillamail.info",
    "sharklasers.com", "guerrillamailblock.com", "grr.la",
    "guerrillamail.co", "spam4.me",

    # --- 10 minute mail family ---
    "10minutemail.com", "10minutemail.net", "10minutemail.org",
    "10minutemail.de", "10minutemail.co.za", "10minutemail.co.uk",
    "10minutemail.us", "10minutemail.info",
    "10minemail.com", "10minemail.net",

    # --- mailinator family ---
    "mailinator.com", "mailinator2.com", "mailinator.net",
    "mailinator.gq", "notmailinator.com", "suremail.info",
    "spamgourmet.com", "spamgourmet.net", "spamgourmet.org",

    # --- yopmail family ---
    "yopmail.com", "yopmail.fr", "yopmail.pp.ua",
    "cool.fr.nf", "jetable.fr.nf", "nospam.ze.tc",
    "nomail.xl.cx", "mega.zik.dj", "speed.1s.fr",
    "courriel.fr.nf", "moncourrier.fr.nf", "monemail.fr.nf",
    "monmail.fr.nf",

    # --- throwam / jetable ---
    "throwam.com", "throwam.net", "jetable.com",
    "jetable.net", "jetable.org", "jetable.pp.ua",
    "nospamfor.us", "nomail.2anom.com",

    # --- temp-mail.org family ---
    "temp-mail.org", "temp-mail.io", "temp-mail.net",
    "temp-mail.ru", "tempmail.com", "tempmail.net",
    "tempmail.org", "tempmail.de", "tempmail.us",
    "tempr.email", "discard.email",

    # --- fakeinbox / mailnesia ---
    "fakeinbox.com", "mailnesia.com", "maildrop.cc",
    "dispostable.com", "disposablemail.es",

    # --- shareable / clix ---
    "clix.pt", "mailtothis.com", "mailismagic.com",
    "mailme.lv", "mailnew.com",

    # --- moakt / spamfree ---
    "moakt.cc", "moakt.co", "moakt.com", "moakt.ws",
    "spamfree24.org", "spamfree24.de", "spamfree24.info",
    "spamfree24.eu", "spamfree24.net",

    # --- discard / throwaway ---
    "throwaway.email", "throam.com", "throwam.net",
    "discardmail.com", "discardmail.de", "safetymail.info",
    "spamgob.com", "spamhereplease.com", "spamhole.com",

    # --- miscellaneous well-known ---
    "mailexpire.com", "filzmail.com", "dispostable.com",
    "cryptogmail.com", "getnada.com", "getairmail.com",
    "owlpic.com", "discard.email", "spamevader.com",
    "mailnull.com", "spamex.com", "spammotel.com",
    "spamgourmet.com", "acmil.com", "mailslapping.com",
    "fakemailgenerator.com", "inoutmail.de", "inoutmail.eu",
    "inoutmail.info", "inoutmail.net", "harakirimail.com",
    "mxfuel.com", "crap.spy.cx", "binkmail.com",
    "bobmail.info", "bobmail.org", "odnorazovoe.ru",
    "temp.emeraldwebmail.com", "noclickemail.com",
    "topranklist.de", "thisisnotmyrealemail.com",
    "fakedemail.com", "fakeemailaddress.org", "fakemail.fr",
    "fakemail.net", "fakemailz.com", "fakeemailgenerator.com",
    "mailpoof.com", "mailseal.de", "mytrashmail.com",
    "nwytg.com", "filzmail.com", "wh4f.org",
    "kurzepost.de", "objectmail.com",
}


def is_disposable_email(email: str) -> bool:
    """Return True if the email's domain is on the disposable blocklist."""
    try:
        domain = email.strip().lower().split("@")[1]
    except IndexError:
        return False
    return domain in DISPOSABLE_EMAIL_DOMAINS
