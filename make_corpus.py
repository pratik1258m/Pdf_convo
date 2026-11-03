"""Seven structurally different PDFs, with the questions they must answer.

The point is genre variety: the engines were tuned on one Selenium lab manual, and
a lab manual is not a policy handbook, an API reference, a paper, a price list or
a Hindi document. Each document here carries its own ground truth next to its own
text, so a question can never drift away from the page it is supposed to be
answered from.

    python3 make_corpus.py            # write corpus/*.pdf and print structure
    python3 test_corpus.py            # score the engines against these questions

Text is authored as flat lines with `|` marking table cell boundaries, then laid
out with real Paragraph and Table flowables so the extracted text has the shape a
downloaded PDF has: wrapped prose, ruled rows, running headers.
"""

import os
from typing import Any, Dict, List, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.styles import TA_LEFT

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")

# macOS ships a Devanagari face whose cmap survives pypdf extraction in logical
# order, which is how machine-generated Hindi PDFs usually arrive.
DEVANAGARI_FONT = "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc"


def _row(line: str) -> List[str]:
    return [cell.strip() for cell in line.split("|")]


class Doc:
    """One corpus PDF: its pages and the questions that must be answered from it."""

    def __init__(self, name: str, title: str, pages: List[List[str]], questions: List[Dict[str, Any]],
                 font: str = "Helvetica", ruled: bool = False, footer: str = ""):
        self.name = name
        self.title = title
        self.pages = pages
        self.questions = questions
        self.font = font
        self.ruled = ruled
        self.footer = footer

    @property
    def path(self) -> str:
        return os.path.join(CORPUS_DIR, f"{self.name}.pdf")


def q(text: str, pages: List[int], contains: List[str] = (), refuse: bool = False,
      not_contains: List[str] = ()) -> Dict[str, Any]:
    """A question with its truth: the page it must come from and the words it must carry.

    `contains` is checked against the answer text case-insensitively. A question
    whose answer is not in the document is marked `refuse` and must be met with
    the out-of-context guardrail, not with a page number.
    """
    return {"q": text, "pages": pages, "contains": list(contains), "refuse": refuse,
            "not_contains": list(not_contains)}


# --------------------------------------------------------------------- 1. lab manual
LAB = Doc(
    "selenium_lab_manual",
    "Automation Laboratory Manual: Web Testing with Selenium",
    pages=[
        [
            "#Automation Laboratory Manual",
            "##Module 1 - Practical 1: Setting Up the Environment",
            "Aim: prepare the Python automation environment for the semester.",
            "Software required: Python 3.11, Google Chrome 120, and Visual Studio Code.",
            "Install the libraries with the command pip install selenium pytest webdriver-manager openpyxl",
            "The driver is created by calling webdriver.Chrome(service=Service(ChromeDriverManager().install())).",
            "Every script must be saved in the folder C:/automation/practicals before execution.",
            "Result: the environment is ready and the sample script runs without errors.",
            "##Module 1 - Practical 2: Login Form Test",
            "Aim: automate the login form of the demo application and verify the page title.",
            "Open the login form with driver.get(\"https://the-internet.herokuapp.com/login\").",
            "Enter the username tomsmith into the field with id user-name.",
            "Enter the password SuperSecretPassword! into the field with id password.",
            "Click the Login button and wait for the flash message to render.",
            "assert \"You logged into a secure area!\" in driver.page_source",
            "Viva question: why is the flash message asserted instead of the URL?",
        ],
        [
            "##Module 1 - Practical 3: Excel Driven Test",
            "Aim: read test data from a workbook and write the verdict back.",
            "Input workbook: students.xlsx with columns name, marks, and grade.",
            "The report is written to results.xlsx in the same folder as the script.",
            "Openpyxl loads the sheet with wb = openpyxl.load_workbook(\"students.xlsx\").",
            "Each row is passed to the test method through pytest.mark.parametrize.",
            "A row fails when marks is below 35, and the reason is stored in the remarks column.",
            "##Module 2 - Practical 4: Wait Handling",
            "Aim: compare implicit, explicit, and hard waits on a slow page.",
            "driver.implicitly_wait(10)",
            "Wait an additional time.sleep(3) before capturing the screenshot.",
            "Set the explicit wait as WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.ID, \"cart\"))).",
            "Observation: the implicit value applies to every find element call in the session.",
            "The timeout value is recorded in the lab record as 15 seconds.",
            "##Module 2 - Practical 5: Alerts and Frames",
            "Aim: handle a JavaScript confirm dialog and an inner iframe.",
            "Switch with driver.switch_to.alert then call alert.accept().",
            "Use alert.dismiss() to reject the dialog without confirming.",
            "Move into the frame with driver.switch_to.frame(\"iframe-result\").",
        ],
        [
            "##Module 3 - Practical 6: Project Report",
            "Aim: run the regression suite and summarise the outcome.",
            "The suite is executed with pytest -v --html=report.html --self-contained-html.",
            "Twelve test cases are written in total, of which nine are automated and three are manual.",
            "Defect status moves from New to Assigned, Verified, and finally Closed.",
            "The report must be signed by the lab instructor before the internal assessment.",
            "Viva question: what is the difference between severity and priority?",
            "Semester evaluation: 30 marks for practicals, 20 for the record, and 50 for the written paper.",
        ],
    ],
    questions=[
        q("What username does the login practical use?", [1], ["tomsmith"]),
        q("Which password is typed into the login form?", [1], ["supersecretpassword"]),
        q("Which website is opened for the login test?", [1], ["the-internet.herokuapp.com"]),
        q("Which packages must be installed for the environment?", [1], ["selenium", "pytest", "webdriver-manager"]),
        q("What input workbook does the excel driven test read?", [2], ["students.xlsx"]),
        q("Which file holds the output of the excel driven test?", [2], ["results.xlsx"]),
        q("What is the explicit wait timeout in practical 4?", [2], ["15"]),
        q("How is the alert dismissed in practical 5?", [2], ["dismiss"]),
        q("What does the login test assert?", [1], ["you logged into a secure area"]),
        q("How many test cases are in the regression suite?", [3], ["twelve"]),
        q("What is the first step of practical 3?", [2], ["students.xlsx", "workbook"]),
        q("Which grade is a fail in the excel driven test?", [2], ["35"]),
        q("What is the penalty for a late submission?", [1, 2, 3], refuse=True),
        q("Who is the prime minister of India?", [1, 2, 3], refuse=True),
    ],
)

# ------------------------------------------------------------------ 2. policy handbook
POLICY = Doc(
    "hr_policy_handbook",
    "People Policy Handbook",
    pages=[
        [
            "#People Policy Handbook",
            "##1 Purpose and Scope",
            "This handbook describes the employment policies that apply to every full-time employee of the "
            "company across all offices in India. It is reviewed once a year by the leadership team and the "
            "human resources business partners, and the version in force is always the one published on the "
            "intranet.",
            "Where a local law is more generous than this handbook, the local law is followed. Nothing in this "
            "document creates a contract of employment, and it should be read together with the appointment "
            "letter and the employment agreement signed at the time of joining.",
            "##2 Employment Types",
            "A probationer is confirmed after six months of satisfactory service, subject to a rating of "
            "meets expectations or better in the probation review.",
            "Contractual staff are engaged for a fixed term of twelve months and may be extended twice before "
            "they are considered for a permanent role.",
            "Interns work for a minimum of eight weeks and are not eligible for leave encashment.",
        ],
        [
            "##3 Leave Policy",
            "3.1 Casual Leave: twelve days are granted at the start of each calendar year and cannot be "
            "accumulated beyond the year in which they are granted.",
            "3.2 Sick Leave: ten days per year, available from the first month of service. Sick leave of more "
            "than three consecutive days requires a medical certificate from a registered practitioner.",
            "3.3 Earned Leave: fifteen days per year, accrued at the rate of one and one fourth days per "
            "completed month. Earned leave may be accumulated up to sixty days and encashed at basic salary.",
            "3.4 Maternity leave follows the Maternity Benefit Act of 2017 and is twenty-six weeks for a first "
            "child. Paternity leave is fifteen calendar days.",
            "Leave must be applied for in the portal at least three working days in advance, except in an "
            "emergency, and the reporting manager approves it within two working days.",
            "##4 Notice Period",
            "An employee below grade 5 serves thirty days of notice, and an employee at grade 5 or above "
            "serves sixty days. Either side may buy out the balance at basic salary, and notice pay is waived "
            "when the resignation is accepted on the last working day itself.",
        ],
        [
            "##5 Reimbursement and Benefits",
            "Travel claims are capped at 4000 rupees per month for local conveyance and require a bill in the "
            "employee's own name. Fuel for a personal vehicle used on company business is reimbursed at the "
            "rate fixed by the finance team each quarter.",
            "Health insurance is a family floater of 500000 rupees covering the employee, spouse, and two "
            "dependent children, with a pre-authorisation desk operational on working days from nine in the "
            "morning to six in the evening.",
            "##6 Code of Conduct",
            "Employees must not accept gifts above 1000 rupees from a vendor without written approval from the "
            "compliance officer. Confidential customer data may not be copied to a personal device under any "
            "circumstance.",
            "##7 Grievance Redressal",
            "A grievance is raised in writing to the human resources business partner, who acknowledges it "
            "within five working days and closes it within thirty. An employee who is not satisfied may "
            "escalate the matter to the Chief Compliance Officer, whose decision is final.",
        ],
    ],
    questions=[
        q("How many days of casual leave are granted each year?", [2], ["twelve"]),
        q("When is a medical certificate required for sick leave?", [2], ["three"]),
        q("How much earned leave can be accumulated?", [2], ["sixty"]),
        q("What is the notice period for an employee at grade 5 or above?", [2], ["sixty"]),
        q("What is the cap on local travel reimbursement?", [3], ["4000"]),
        q("What is the health insurance cover for a family floater?", [3], ["500000"]),
        q("Who does an employee escalate a grievance to?", [3], ["compliance officer"]),
        q("Within how many days must the HR business partner acknowledge a grievance?", [3], ["five"]),
        q("After how many months is a probationer confirmed?", [1], ["six"]),
        q("What is the maximum maternity leave?", [2], ["twenty-six"]),
        q("List the sections of this handbook.", [1, 2, 3], ["leave", "notice", "conduct"]),
        q("Summarise the reimbursement page.", [3], ["travel", "insurance"]),
        q("What is the company's stock ticker?", [1, 2, 3], refuse=True),
    ],
)

# ---------------------------------------------------------------------- 3. API reference
API = Doc(
    "inventory_api_reference",
    "StockLite Inventory API Reference",
    pages=[
        [
            "#StockLite Inventory API Reference",
            "##Authentication",
            "Every request carries the key in the X-Api-Key header; keys are issued per environment and are "
            "never sent in the query string. The base URL for production is https://api.stocklite.example/v1.",
            "##Rate limits",
            "An API key may issue 100 requests per minute. A request that exceeds the limit is answered with "
            "HTTP 429 and a Retry-After header of 30 seconds.",
            "##GET /items",
            "Returns the items in a warehouse, one page at a time.",
            "Name|Type|Required|Default|Description",
            "warehouse_id|string|yes|-|Identifier of the warehouse to read",
            "limit|integer|no|50|Number of items per page, maximum 200",
            "cursor|string|no|-|Opaque token from the previous response",
            "updated_since|date|no|-|ISO 8601 date, filters items changed after it",
            "The response is application/json and the page envelope carries data, next_cursor, and has_more.",
        ],
        [
            "##POST /items",
            "Creates an item. The body is JSON and the only mandatory field is sku.",
            "Name|Type|Required|Default|Description",
            "sku|string|yes|-|Stock keeping unit, unique per warehouse",
            "name|string|yes|-|Human readable label shown in reports",
            "quantity|integer|no|0|On-hand count at the time of creation",
            "unit_price|number|no|0.00|Price in the warehouse currency, two decimals",
            "tags|array|no|empty list|Searchable labels attached to the item",
            "A duplicate sku is answered with HTTP 409 and the error code sku_already_exists.",
            "##DELETE /items/{sku}",
            "Removes an item that has never been part of a movement. An item with history returns HTTP 409 "
            "with the error code item_has_movements and must be archived instead.",
            "##Webhooks",
            "Set the callback URL in the console. StockLite posts item.updated whenever quantity changes, and "
            "signs the body with HMAC SHA-256 in the X-StockLite-Signature header.",
            "Delivery is retried 5 times with exponential backoff and then marked failed.",
        ],
        [
            "##Error codes",
            "Code|Meaning|Action",
            "400|bad_request|Fix the JSON body; the message names the offending field",
            "401|unauthorized|The key is missing or revoked, issue a new one in the console",
            "404|not_found|The sku or warehouse does not exist in this environment",
            "409|conflict|The resource already exists or cannot be deleted",
            "429|rate_limited|Back off for the number of seconds in Retry-After",
            "500|internal_error|Safe to retry once; StockLite pages on-call above a one minute rate",
            "##Versioning",
            "The major version is in the path and never changes without twelve months of notice. Minor "
            "additions are additive and are announced in the changelog.",
            "The sandbox environment is https://sandbox.stocklite.example/v1 and holds 90 days of data.",
            "Pagination uses a cursor rather than an offset, and the last page has has_more set to false.",
        ],
    ],
    questions=[
        q("Which header carries the API key?", [1], ["x-api-key"]),
        q("What is the production base URL?", [1], ["api.stocklite.example"]),
        q("How many requests per minute are allowed?", [1], ["100"]),
        q("What does HTTP 429 mean and what should the client do?", [1, 3], ["retry-after", "back off"]),
        q("Which field is mandatory when creating an item?", [2], ["sku"]),
        q("What is the default page limit on GET /items?", [1], ["50"]),
        q("What is the maximum value of the limit parameter?", [1], ["200"]),
        q("Which HTTP code is returned for a duplicate sku?", [2], ["409"]),
        q("How is the webhook body signed?", [2], ["hmac"]),
        q("How many times is webhook delivery retried?", [2], ["5"]),
        q("What does the error code item_has_movements mean?", [2], ["archiv"]),
        q("Which endpoint removes an item that has never moved?", [2], ["delete"]),
        q("What is the sandbox URL?", [3], ["sandbox.stocklite.example"]),
        q("How much notice does a major version change require?", [3], ["twelve"]),
        q("What is the refund policy for an item?", [1, 2, 3], refuse=True),
    ],
    ruled=True,
)

# ---------------------------------------------------------------------- 4. research paper
PAPER = Doc(
    "retrieval_research_paper",
    "Hybrid Retrieval for Long Document Question Answering",
    pages=[
        [
            "#Hybrid Retrieval for Long Document Question Answering",
            "##Abstract",
            "Retrieval systems that answer questions about a single long document usually rank passages with "
            "BM25 alone. We introduce ConvoQA, a dataset of 1842 human questions written over 96 technical "
            "PDFs, and a hybrid ranker that combines Okapi BM25 with an idf-weighted coverage gate. On the "
            "dev split the hybrid ranker reaches an F1 of 0.82 against 0.74 for BM25 alone, while adding no "
            "model download to the serving path.",
            "##1 Introduction",
            "Spoken assistants raise the bar for retrieval: a wrong passage is heard as a confident lie, and a "
            "slow answer is heard as a broken device. We therefore measure retrieval by the evidence it puts "
            "in front of the answerer rather than by ranking metrics alone.",
            "Prior work on open-domain QA optimises recall at fifty passages. Our setting is the opposite: "
            "four passages, one document, and one second of wall clock budget.",
        ],
        [
            "##2 The ConvoQA Dataset",
            "Questions were written by six annotators against public laboratory manuals, policy handbooks and "
            "vendor datasheets. Each question is paired with the page and the exact sentence that answers it. "
            "Disagreement between annotators was resolved by a third reader, and 41 questions were discarded "
            "because no passage in the source PDF answered them.",
            "The split is 80 percent training, 20 percent evaluation, with no document shared between them.",
            "##3 Method",
            "Passages are formed by a line-aware chunker that keeps a section heading with the lines under it "
            "and targets sixty words. Ranking is Okapi BM25 with k1 of 1.5 and b of 0.75 over two fields, body "
            "and heading, where the heading score is multiplied by 1.4.",
            "A candidate is accepted only when the idf-weighted coverage of the query terms in the best "
            "passage exceeds 0.34. Coverage uses only terms that occur somewhere in the document, so a long "
            "question is not penalised for vocabulary the document never had.",
            "##4 Results",
            "Table 1|Ranker|Precision at 4|Answer F1|Median latency",
            "BM25|0.69|0.74|4 ms",
            "BM25 plus heading field|0.75|0.78|4 ms",
            "Hybrid with coverage gate|0.81|0.82|5 ms",
            "Cross encoder reranker|0.86|0.85|214 ms",
            "The coverage gate is responsible for most of the gain at no measurable cost, and the reranker "
            "buys four points of F1 for two orders of magnitude of latency.",
        ],
        [
            "##5 Ablation",
            "Removing the heading field costs 0.04 F1. Replacing line-aware chunks with fixed windows of 500 "
            "characters costs 0.03 F1 and breaks code quotation, because a single statement is no longer a "
            "retrievable unit. Removing ordinal expansion costs 0.02 F1 on questions that name a section.",
            "##6 Conclusion and Future Work",
            "A single long document can be answered well with a hand-written ranker. Future work covers "
            "spoken barge-in, where the assistant must stop mid-sentence, and multi-document retrieval over a "
            "shared corpus. We release ConvoQA and the chunker under the MIT licence.",
            "##References",
            "Robertson and Zaragoza. The probabilistic foundation for vector-space retrieval. SIGIR 2009.",
            "Karpukhin et al. Dense Passage Retrieval for Open-Domain Question Answering. EMNLP 2020.",
            "Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers. NAACL 2019.",
        ],
    ],
    questions=[
        q("What dataset was introduced in this paper?", [1], ["convoqa"]),
        q("How many questions are in ConvoQA?", [1], ["1842"]),
        q("How many documents does ConvoQA cover?", [1], ["96"]),
        q("What answer F1 does the hybrid ranker reach?", [1], ["0.82"]),
        q("How many annotators wrote the questions?", [2], ["six"]),
        q("What train to evaluation split was used?", [2], ["80"]),
        q("Which BM25 parameters were chosen?", [2], ["1.5", "0.75"]),
        q("What is the coverage threshold for accepting a candidate?", [2], ["0.34"]),
        q("How much latency does the cross encoder reranker add?", [2], ["214"]),
        q("What does removing the heading field cost?", [3], ["0.04"]),
        q("What future work is planned?", [3], ["barge-in", "multi-document"]),
        q("Under which licence is ConvoQA released?", [3], ["mit"]),
        q("Who won the 2022 world cup?", [1, 2, 3], refuse=True),
    ],
    ruled=True,
)

# ----------------------------------------------------------------------- 5. price list
PRICES = Doc(
    "inverter_price_list",
    "Q4 Dealer Price List: Solar Inverters",
    pages=[
        [
            "#Q4 Dealer Price List: Solar Inverters",
            "Prices are in Indian rupees, include GST at 12 percent, and exclude installation. Effective from "
            "1 October until the next printed list supersedes it.",
            "Item|SKU|Price|Warranty|Stock",
            "APDCR RCT-6000D 48V pure sine wave inverter|RCT-6000|12499|2 years|in stock",
            "Luminous NXG+ 5520 48V inverter|NXG5520|14990|2 years|in stock",
            "Microtek UPS 2470EF elliptical sine wave|2470EF|9850|1 year|low stock",
            "Sukam ROBO 4024MPPT hybrid inverter|4024MPPT|21300|5 years|in stock",
            "Fermator I-Max 1250VA single phase|IMAX1250|6120|2 years|discontinued",
            "The RCT-6000 supports a 6000 VA connected load and a 48 battery bank.",
            "Warranty is counted from the invoice date and requires the dealer stamp.",
        ],
        [
            "##Batteries and accessories",
            "Item|SKU|Price|Warranty|Stock",
            "Tubular 150Ah 12V deep cycle battery|TUB150|13900|36 months|in stock",
            "Tubular 200Ah 12V C100 battery|TUB200|17450|48 months|in stock",
            "Solar charge controller 60A MPPT|MPPT60|4299|2 years|in stock",
            "DC surge protection device Type 2|SPD2|1180|1 year|in stock",
            "Battery copper terminal set 200Ah|TERM200|340|6 months|in stock",
            "A complete 5.5 kW off-grid package with two 200Ah batteries is quoted at 58900 rupees.",
            "Installation of a hybrid inverter is charged separately at 2200 rupees per unit.",
        ],
    ],
    questions=[
        q("What is the price of the APDCR RCT-6000D?", [1], ["12499"]),
        q("Which inverter carries a five year warranty?", [1], ["sukam", "4024mppt"]),
        q("How much does the Luminous NXG+ 5520 cost?", [1], ["14990"]),
        q("Which item is discontinued?", [1], ["imax1250", "fermator"]),
        q("What is the price of the 200Ah tubular battery?", [2], ["17450"]),
        q("How much is a complete 5.5 kW package with two batteries?", [2], ["58900"]),
        q("What does installation of a hybrid inverter cost?", [2], ["2200"]),
        q("What GST rate is included in these prices?", [1], ["12"]),
        q("Which item has the lowest warranty period?", [1, 2], ["340", "term200", "6 months"]),
        q("Is the MPPT60 controller in stock?", [2], ["in stock"]),
        q("What is the exchange rate of the dollar?", [1, 2], refuse=True),
    ],
    ruled=True,
)

# --------------------------------------------------------------------- 6. short checklist
CHECKLIST = Doc(
    "lab_safety_checklist",
    "Laboratory Safety Checklist",
    pages=[
        [
            "#Laboratory Safety Checklist",
            "##Before the first run",
            "1. Wear the lab coat and closed shoes.",
            "2. Tie back long hair.",
            "3. Locate the mains switch.",
            "4. Check the earthing.",
            "5. Report a damaged cable.",
            "##Definitions",
            "Earthing: connecting the chassis of the equipment to the ground so that a fault current has a low "
            "resistance path.",
            "Mercury thermometer: prohibited in the lab since 2019.",
            "Static charge: discharge by touching a grounded metal plate before handling a module.",
            "##Rules",
            "No food inside the lab.",
            "No mobile at the bench.",
            "Sign the register.",
            "Two students per machine.",
            "Report a burn at once.",
        ],
        [
            "##Sign-off",
            "All practicals must be signed by the lab instructor before submission.",
            "The record is verified in the last week of the semester.",
            "A student who has not completed eight practicals is not admitted to the internal assessment.",
            "Contact: lab instructor, room 214, extension 305.",
        ],
    ],
    questions=[
        q("What is earthing?", [1], ["chassis", "ground"]),
        q("Is a mobile phone allowed at the bench?", [1], ["no mobile at the bench"]),
        q("Are thermometers banned?", [1], ["mercury", "prohibited"]),
        q("Who signs the practical record?", [2], ["lab instructor"]),
        q("How many practicals must be completed before the internal assessment?", [2], ["eight"]),
        q("What is the extension number of the lab instructor?", [2], ["305"]),
        q("Which shoes are required?", [1], ["closed"]),
        q("Where is the lab instructor's room?", [2], ["214"]),
        q("List the steps to take before the first run.", [1], ["lab coat", "earthing", "mains"]),
    ],
)

# ------------------------------------------------------------------------ 7. Hindi manual
HINDI = Doc(
    "hindi_lab_manual",
    "सॉफ्टवेयर परीक्षण प्रयोगशाला पुस्तिका",
    font=DEVANAGARI_FONT,
    pages=[
        [
            "#सॉफ्टवेयर परीक्षण प्रयोगशाला पुस्तिका",
            "##माड्यूल 1 - प्रैक्टिकल 1: पर्यावरण सेटअप",
            "उद्देश्य: सेमेस्टर के लिए ऑटोमेशन पर्यावरण तैयार करना।",
            "पाइथन 3.11 और गूगल क्रोम 120 स्थापित करें।",
            "लाइब्रेरी इस कमांड से इंस्टॉल करें pip install selenium pytest openpyxl",
            "##माड्यूल 1 - प्रैक्टिकल 2: लॉगिन परीक्षण",
            "उद्देश्य: डेमो लॉगिन फॉर्म को स्वचालित करना।",
            "यूज़रनेम tomsmith और पासवर्ड SuperSecretPassword! दर्ज करें।",
            "ड्राइवर https://the-internet.herokuapp.com/login खोलता है।",
            "लॉगिन के बाद पृष्ठ शीर्षक The Internet होना चाहिए।",
        ],
        [
            "##माड्यूल 2 - प्रैक्टिकल 3: एक्सेल सहायित परीक्षण",
            "उद्देश्य: कार्यपुस्तिका से डेटा पढ़ना और परिणाम लिखना।",
            "इनपुट फाइल students.xlsx है जिसमें नाम, अंक और श्रेणी के स्तंभ हैं।",
            "परिणाम results.xlsx फाइल में लिखा जाता है।",
            "जिस विद्यार्थी के अंक 35 से कम हैं वह अनुत्तीर्ण माना जाएगा।",
            "##माड्यूल 2 - प्रैक्टिकल 4: प्रतीक्षा नियंत्रण",
            "उद्देश्य: अंतर्निहित और स्पष्ट प्रतीक्षा की तुलना करना।",
            "ड्राइवर का इम्प्लीसिट वेट 10 सेकंड सेट करें।",
            "स्पष्ट वेट का मान 15 सेकंड है और हार्ड वेट 3 सेकंड है।",
            "रैंडम आईडी में छह अंक होते हैं और अंतर्निहित स्लीप प्रतिबंधित है।",
        ],
    ],
    questions=[
        q("लॉगिन के लिए यूज़रनेम क्या है?", [1], ["tomsmith"]),
        q("पासवर्ड कौन सा दर्ज करना है?", [1], ["supersecretpassword"]),
        q("कौन सी वेबसाइट खोली जाती है?", [1], ["the-internet.herokuapp.com"]),
        q("एक्सेल परीक्षण में इनपुट फाइल कौन सी है?", [2], ["students.xlsx"]),
        q("कितने अंक से कम होने पर विद्यार्थी अनुत्तीर्ण है?", [2], ["35"]),
        q("इम्प्लीसिट वेट कितने सेकंड का है?", [2], ["10"]),
        q("स्पष्ट वेट का मान क्या है?", [2], ["15"]),
        q("कौन सी लाइब्रेरी इंस्टॉल करनी है?", [1], ["selenium"]),
        q("प्राइस लिस्ट में क्या दरें दी गई हैं?", [1, 2], refuse=True),
    ],
)

DOCUMENTS: List[Doc] = [LAB, POLICY, API, PAPER, PRICES, CHECKLIST, HINDI]


def _styles(font: str) -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["BodyText"], fontName=font, fontSize=9.6, leading=13,
                          alignment=TA_LEFT, spaceAfter=3)
    heading = ParagraphStyle("heading", parent=base["Heading2"], fontName=font, fontSize=12, leading=15,
                             spaceBefore=9, spaceAfter=4)
    title = ParagraphStyle("title", parent=base["Title"], fontName=font, fontSize=16, leading=20, spaceAfter=6)
    return {"body": body, "heading": heading, "title": title}


def _table(rows: List[List[str]], font: str, ruled: bool) -> Table:
    table = Table(rows)
    grid = ("GRID", (0, 0), (-1, -1), 0.4, colors.Color(0.75, 0.75, 0.75)) if ruled else \
           ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.955, 0.955, 0.955)]),
        grid,
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return table


def _table_run(page: List[str], start: int) -> int:
    """One past the last `|` row of the block that begins at `start`."""
    end = start
    while end < len(page) and "|" in page[end]:
        end += 1
    return end


def build(doc: Doc) -> str:
    if doc.font == DEVANAGARI_FONT and doc.font not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("Deva", DEVANAGARI_FONT))
    font = "Deva" if doc.font == DEVANAGARI_FONT else "Helvetica"
    styles = _styles(font)
    os.makedirs(CORPUS_DIR, exist_ok=True)
    story: List[Any] = []
    for index, page in enumerate(doc.pages):
        if index:
            story.append(PageBreak())
        cursor = 0
        while cursor < len(page):
            line = page[cursor]
            if "|" in line:
                stop = _table_run(page, cursor)
                story.append(_table([_row(row) for row in page[cursor:stop]], font, doc.ruled))
                cursor = stop
                continue
            style = styles["title"] if line.startswith("#") and not line.startswith("##") else \
                styles["heading"] if line.startswith("##") else styles["body"]
            story.append(_paragraph(line.lstrip("#"), style, font))
            cursor += 1
        story.append(Spacer(1, 6))
    pdf = SimpleDocTemplate(doc.path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title=doc.title,
                            author="corpus generator", subject=doc.title)
    pdf.build(story)
    return doc.path


def _paragraph(text: str, style: ParagraphStyle, font: str):
    from reportlab.platypus import Paragraph
    return Paragraph(text, style)


def structure() -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    for doc in DOCUMENTS:
        out.append((doc.name, {"pages": len(doc.pages), "questions": len(doc.questions),
                               "rows": sum(len(p) for p in doc.pages)}))
    return out


if __name__ == "__main__":
    total_q = 0
    for document in DOCUMENTS:
        path = build(document)
        size = os.path.getsize(path)
        print(f"  wrote {os.path.basename(path):<34} {size:>7}B  {len(document.pages)} pages  "
              f"{len(document.questions)} questions")
        total_q += len(document.questions)
    print(f"\n{len(DOCUMENTS)} PDFs in {CORPUS_DIR}, {total_q} questions with ground truth.")
