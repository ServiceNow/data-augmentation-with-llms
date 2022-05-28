"""
All banking77 specific utilities are implemented here.
"""
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from haven import haven_utils as hu
import os

INTENTS = [
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]


class Banking77:
    def __init__(self, name):
        self.data_name = name
        self.full_path = f"./data/{name}/full/dataset.pkl"
        self.num_examples = 10

        # 1. save dataset.pkl
        if not os.path.exists(self.full_path):
            self.dataset = load_dataset(name).rename_column("label", "intent")
            # get the splits
            train, validation = train_test_split(
                self.dataset["train"],
                train_size=0.9,
                stratify=self.dataset["train"]["intent"],
            )

            # get data dict
            data_dict = {
                "train": train,
                "val": validation,
                "test": self.dataset["test"].to_dict(),
            }
            hu.save_pkl(self.full_path, data_dict)
        else:
            data_dict = hu.load_pkl(self.full_path)

        # 2. Group by Intent
        name2id = {k: i for i, k in enumerate(INTENTS)}
        hu.save_json(f"./data/{name}/name2id.json", name2id)
        id2name = {i: k for i, k in enumerate(INTENTS)}
        hu.save_json(f"./data/{name}/id2name.json", id2name)

        # intents = list(name2id.keys())

        self.dataset_by_intent = {}
        for split in ["train", "val", "test"]:
            self.dataset_by_intent[split] = {}
            text_intent_dict = data_dict[split]
            text_list, intent_list = (
                text_intent_dict["text"],
                map(lambda x: id2name[x], text_intent_dict["intent"]),
            )

            # get texts from intent
            intent2texts = {}
            for t, i in zip(text_list, intent_list):
                if i not in intent2texts:
                    intent2texts[i] = []
                intent2texts[i] += [t]
            self.dataset_by_intent[split] = intent2texts

        self.domain_to_intent = {name: INTENTS}
        self.gpt3_batch_size: int = 128

    def parse_and_load(self):
        return self.dataset_by_intent
