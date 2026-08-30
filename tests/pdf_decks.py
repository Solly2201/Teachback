"""Slide-deck PDF fixtures used by the ingestion regression tests.

``cloud_deck()`` deliberately reproduces the deck that broke the old pipeline:
a running header, a running footer, slide numbers, an "(c) Copyright 2014 EMC
Corporation. All rights reserved." line on every slide, a section-divider
slide, a slide that carries nothing but boilerplate, and — importantly — a
legitimate concept ("Cloud Computing") that genuinely recurs as a slide title.

``sorting_deck()`` is a completely different subject with the same structural
problems, so the tests can assert that nothing in the pipeline is tuned to one
particular deck.
"""
from pdf_fixture import build_pdf, line

HEADER = "Cloud Infrastructure and Services"
FOOTER = "Module 1: Introduction to Cloud Computing"
COPYRIGHT = "(c) Copyright 2014 EMC Corporation. All rights reserved."


def _chrome(page_no: int, header: str = HEADER, footer: str = FOOTER,
            copyright_line: str = COPYRIGHT) -> list[dict]:
    """The decoration every slide of a corporate deck carries."""
    return [
        line(header, 16, 9),
        line(footer, 500, 8),
        line(str(page_no), 500, 8, x=670),
        line(copyright_line, 516, 8),
    ]


def cloud_deck() -> bytes:
    pages = [
        # 1 — title slide
        _chrome(1) + [
            line("Cloud Computing", 150, 34, bold=True),
            line("Lesson: Cloud Computing Overview", 210, 18),
        ],
        # 2 — a definition slide with real explanatory prose
        _chrome(2) + [
            line("Cloud Computing", 50, 26, bold=True),
            line("Cloud computing is a model that enables on-demand network access to a", 120, 14),
            line("shared pool of configurable computing resources such as servers, storage", 140, 14),
            line("and applications.", 160, 14),
            line("These resources can be provisioned and released quickly with very little", 190, 14),
            line("management effort.", 210, 14),
        ],
        # 3 — a section divider: heading only, no teaching content
        _chrome(3) + [
            line("Essential Cloud Characteristics", 200, 30, bold=True),
        ],
        # 4 — a real concept with a definition and a bullet list
        _chrome(4) + [
            line("On-demand Self-service", 50, 26, bold=True),
            line("Consumers can provision computing resources automatically without", 120, 14),
            line("requiring human interaction with each service provider.", 140, 14),
            line("The provisioning happens through a portal or an API.", 170, 14),
        ],
        # 5 — another real concept
        _chrome(5) + [
            line("Broad Network Access", 50, 26, bold=True),
            line("Broad network access means the capabilities are available over the", 120, 14),
            line("network and reached through standard mechanisms.", 140, 14),
            line("Clients include laptops, tablets and mobile phones.", 170, 14),
        ],
        # 6 — a concept taught with bullets rather than prose
        _chrome(6) + [
            line("Service Models", 50, 26, bold=True),
            line("A service model describes how much of the stack the provider manages", 115, 14),
            line("and how much the customer keeps.", 135, 14),
            line("• IaaS provides virtual machines, storage and networks.", 175, 14),
            line("• PaaS provides a managed platform for running applications.", 200, 14),
            line("• SaaS delivers finished applications over the network.", 225, 14),
        ],
        # 7 — an example slide
        _chrome(7) + [
            line("Measured Service", 50, 26, bold=True),
            line("Cloud systems automatically meter resource use so that usage can be", 120, 14),
            line("monitored, controlled and reported to the customer.", 140, 14),
            line("Example: a customer is billed for the number of GB-hours of storage used.", 180, 14),
        ],
        # 8 — a slide that carries nothing but boilerplate
        _chrome(8),
        # 9 — the legitimately repeated concept, taught again in context
        _chrome(9) + [
            line("Cloud Computing", 50, 26, bold=True),
            line("Cloud computing changes capital expenditure into operating expenditure", 120, 14),
            line("because organizations rent capacity instead of buying hardware.", 140, 14),
        ],
        # 10 — a closing slide made only of navigation text
        _chrome(10) + [
            line("Thank You", 200, 30, bold=True),
            line("www.example-training.com", 260, 12),
        ],
    ]
    return build_pdf(pages)


def sorting_deck() -> bytes:
    header = "CS201 Data Structures"
    footer = "Chapter 4: Sorting"
    notice = "(c) 2019 University Press. All rights reserved."
    pages = [
        _chrome(1, header, footer, notice) + [
            line("Sorting Algorithms", 150, 34, bold=True),
        ],
        _chrome(2, header, footer, notice) + [
            line("Bubble Sort", 50, 26, bold=True),
            line("Bubble sort repeatedly compares neighbouring elements and swaps them", 120, 14),
            line("when they are in the wrong order.", 140, 14),
            line("Each pass moves the largest remaining element to the end of the list.", 170, 14),
            line("Example: [3, 1, 2] becomes [1, 2, 3] after two passes.", 200, 14),
        ],
        _chrome(3, header, footer, notice) + [
            line("Merge Sort", 50, 26, bold=True),
            line("Merge sort splits the list in half, sorts each half and then merges the", 120, 14),
            line("two sorted halves back together.", 140, 14),
            line("The merge step compares the front of each half and takes the smaller value.", 170, 14),
        ],
        _chrome(4, header, footer, notice) + [
            line("Time Complexity", 50, 26, bold=True),
            line("Time complexity describes how the running time grows as the input grows.", 120, 14),
            line("Bubble sort runs in quadratic time while merge sort runs in n log n time.", 150, 14),
        ],
        _chrome(5, header, footer, notice),
    ]
    return build_pdf(pages)


def scanned_deck() -> bytes:
    """An image-only PDF: pages exist, almost no selectable text."""
    return build_pdf([[line(".", 200, 10)] for _ in range(6)])
