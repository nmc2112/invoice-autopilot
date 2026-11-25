"""Synthetic French supplier invoices with ground truth (fictional suppliers only).

Each invoice comes with the expected extracted fields and the expected expense
account from the French chart of accounts (Plan Comptable Général). A few
invoices are deliberately inconsistent so the control checks can be tested.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

# account -> (label, VAT rate, [(fictional supplier, [descriptions])])
CATALOG: dict[str, tuple[str, float, list[tuple[str, list[str]]]]] = {
    "606100": ("Fournitures non stockables (eau, énergie)", 20.0, [
        ("Énergie Plus SAS", ["Consommation électricité bureaux", "Abonnement et consommation électrique"]),
        ("Gaz du Centre", ["Fourniture de gaz naturel", "Consommation gaz chauffage locaux"]),
        ("Eaux de Seine Services", ["Consommation d'eau trimestrielle", "Abonnement eau et assainissement"]),
    ]),
    "606400": ("Fournitures administratives", 20.0, [
        ("BureauPro Distribution", ["Ramettes papier A4 et classeurs", "Stylos, agrafes et enveloppes"]),
        ("PapeterieNet", ["Cartouches d'encre imprimante", "Fournitures de bureau diverses"]),
    ]),
    "613200": ("Locations immobilières", 20.0, [
        ("Espaces Pro Immobilier", ["Loyer bureaux mensuel", "Loyer et charges locatives"]),
        ("CoWork Bastille", ["Location postes de travail coworking", "Location salle de réunion"]),
    ]),
    "615600": ("Maintenance", 20.0, [
        ("TechMaint Services", ["Contrat de maintenance informatique", "Intervention dépannage serveur"]),
        ("Ascenseurs Lumière", ["Maintenance annuelle ascenseur", "Visite technique ascenseur"]),
        ("Clim'Expert", ["Entretien climatisation", "Maintenance préventive chauffage"]),
    ]),
    "622600": ("Honoraires", 20.0, [
        ("Cabinet Durand Avocats", ["Honoraires conseil juridique", "Rédaction contrat commercial"]),
        ("Audit & Chiffres Associés", ["Honoraires expertise comptable", "Mission révision des comptes"]),
    ]),
    "625100": ("Voyages et déplacements", 10.0, [
        ("RailExpress Voyages", ["Billet de train Paris-Lyon", "Aller-retour train Paris-Lille"]),
        ("Hôtel Le Central", ["Nuitée hôtel déplacement client", "Hébergement séminaire 2 nuits"]),
        ("TaxiCity", ["Course taxi aéroport", "Transport taxi rendez-vous client"]),
    ]),
    "626000": ("Frais postaux et télécommunications", 20.0, [
        ("TéléFibre Pro", ["Abonnement fibre internet", "Forfait téléphonie mobile flotte"]),
        ("Courrier Express", ["Envoi recommandé et affranchissement", "Frais d'expédition colis"]),
    ]),
    "623100": ("Annonces et insertions", 20.0, [
        ("AdNet Digital", ["Campagne publicitaire en ligne", "Achat d'espaces publicitaires réseaux sociaux"]),
        ("ImprimFlyers", ["Impression flyers promotionnels", "Impression affiches salon"]),
    ]),
}

MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
          "septembre", "octobre", "novembre", "décembre"]


@dataclass
class SyntheticInvoice:
    invoice_id: str
    text: str
    supplier: str
    invoice_number: str
    invoice_date: str
    amount_ht: float
    vat_rate: float
    vat_amount: float
    amount_ttc: float
    account: str
    description: str
    expected_issue: str | None = None   # "amount_mismatch", "above_limit", "duplicate" or None


def _fmt(x: float) -> str:
    return f"{x:,.2f}".replace(",", " ").replace(".", ",")


def _render(inv: SyntheticInvoice, rng: random.Random, shown_ttc: float) -> str:
    d = date.fromisoformat(inv.invoice_date)
    date_str = rng.choice([d.strftime("%d/%m/%Y"), f"{d.day} {MONTHS[d.month - 1]} {d.year}"])
    ht_label, ttc_label = rng.choice([("Total HT", "Total TTC"), ("Montant HT", "Net à payer TTC"),
                                      ("Sous-total HT", "Montant TTC")])
    return "\n".join([
        inv.supplier.upper(),
        f"SIRET {rng.randint(10**13, 10**14 - 1)} - TVA intracom. FR{rng.randint(10**10, 10**11 - 1)}",
        "",
        f"FACTURE N° {inv.invoice_number}",
        f"Date : {date_str}",
        "Client : ACME CONSEIL SAS, 12 rue de la Paix, 75002 Paris",
        "",
        "Désignation                                   Montant",
        f"{inv.description:<45} {_fmt(inv.amount_ht)} EUR",
        "",
        f"{ht_label} : {_fmt(inv.amount_ht)} EUR",
        f"TVA {str(inv.vat_rate).replace('.0', '')} % : {_fmt(inv.vat_amount)} EUR",
        f"{ttc_label} : {_fmt(shown_ttc)} EUR",
        "Échéance : 30 jours",
        "Document de test - données fictives",
    ])


def generate_invoices(n: int = 200, seed: int = 0, year: int = 2025,
                      issue_rate: float = 0.08) -> list[SyntheticInvoice]:
    rng = random.Random(seed)
    invoices: list[SyntheticInvoice] = []
    for i in range(n):
        account = rng.choice(list(CATALOG))
        _, rate, suppliers = CATALOG[account]
        supplier, descriptions = rng.choice(suppliers)
        ht = round(rng.uniform(40, 4000), 2)
        issue = None
        r = rng.random()
        if r < issue_rate / 3:
            issue = "above_limit"
            ht = round(rng.uniform(12000, 40000), 2)
        vat = round(ht * rate / 100, 2)
        ttc = round(ht + vat, 2)
        inv = SyntheticInvoice(
            invoice_id=f"INV-{seed}-{i:04d}", text="", supplier=supplier,
            invoice_number=f"{supplier[:3].upper()}-{year}-{rng.randint(1000, 99999)}",
            invoice_date=(date(year, 1, 1) + timedelta(days=rng.randint(0, 364))).isoformat(),
            amount_ht=ht, vat_rate=rate, vat_amount=vat, amount_ttc=ttc,
            account=account, description=rng.choice(descriptions), expected_issue=issue,
        )
        shown_ttc = ttc
        if issue is None and issue_rate / 3 <= r < 2 * issue_rate / 3:
            inv.expected_issue = "amount_mismatch"
            shown_ttc = round(ttc + rng.choice([-1, 1]) * rng.uniform(5, 50), 2)
            inv.amount_ttc = shown_ttc   # the invoice really shows a wrong total
        inv.text = _render(inv, rng, shown_ttc)
        invoices.append(inv)

    # Duplicates: the same invoice received twice.
    for j in range(max(1, int(n * issue_rate / 3))):
        original = rng.choice([x for x in invoices if x.expected_issue is None])
        dup = SyntheticInvoice(**{**asdict(original), "invoice_id": f"INV-{seed}-DUP{j}",
                                  "expected_issue": "duplicate"})
        invoices.insert(rng.randint(invoices.index(original) + 1, len(invoices)), dup)
    return invoices


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic supplier invoices")
    parser.add_argument("--out", default="data/invoices.jsonl")
    parser.add_argument("-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for inv in generate_invoices(args.n, args.seed):
            fh.write(json.dumps(asdict(inv), ensure_ascii=False) + "\n")
    print(f"Wrote {args.n}+ invoices to {out}")


if __name__ == "__main__":
    main()
