### **Ancient Script AI – Sanskrit OCR - Translation Pipeline Overview**

Ancient Script AI is an end-to-end system for extracting, normalizing, and translating Sanskrit text from manuscript images into meaningful English.

* The project focuses on classical Sanskrit manuscripts and inscriptions, combining:
* High-accuracy OCR

* Sanskrit-aware normalization
* Neural Machine Translation fine-tuned on large Sanskrit–English corpora

* The system is designed to be academically reliable, linguistically grounded, and production-extensible.

**Key Features**

* Sanskrit-specific normalization (danda handling, diacritics, symbols)
* Fine-tuned IndicTrans2 (Sanskrit → English) translation model

* Verse-wise translation to reduce repetition and hallucination
* Modular pipeline (OCR → Normalize → Translate)

* Ready for UI and API integration

#### **Pipeline Flow**

* Image Input
* Manuscript or inscription image

* OCR Extraction
* Google Vision OCR

* Bounding boxes used for line ordering
* Sanskrit Normalization

* Danda and double-danda normalization
* Preservation of sacred symbols (e.g., ॐ)

* Removal of OCR noise
* Verse segmentation

* Neural Translation
* Fine-tuned IndicTrans2 model

* Sanskrit (san_Deva) → English (eng_Latn)
* Verse-wise translation

* Final Output

* Clean Sanskrit text
* Meaningful English translation

**Model Details Translation Model**

* Base: IndicTrans2 (ai4bharat)
* Task: Sanskrit → English

* Script: Devanagari
* Fine-tuned on: Itihāsa and classical Sanskrit texts, Parallel Sanskrit–English corpora

**Training size:**

* ~75k training sentences

* ~6k validation
* ~11k test

**Why IndicTrans2**

* Indic-language optimized
* Script-aware tokenization

* Superior handling of Sanskrit morphology compared to generic models

**Current Status**

* OCR: Production-ready
* Translation model: Fine-tuned and stable

* Sanskrit normalization: Implemented (ongoing improvements)
* UI: Planned

* API: Planned

**Limitations**

* OCR quality depends on image clarity
* Verse-level semantic restructuring is rule-based

* Tamil and other scripts are not yet supported
* UI not included in current version

**Roadmap**

* Advanced Sanskrit syntactic normalization
* Dictionary-guided translation constraints

* REST API using FastAPI
* Web UI (HTML → React)

* Support for Tamil inscriptions (future phase)

**Intended Users**

* Indologists and Sanskrit scholars
* **Archaeologists** and **epigraphists**

* Digital humanities researchers
* Cultural heritage projects

* Academic institutions

**License**

* This project is intended for research and educational use. Commercial usage requires appropriate licensing for OCR and model components.

**Acknowledgements**

* AI4Bharat – IndicTrans2
* Google Cloud Vision OCR

* Classical Sanskrit corpora and lexicons
