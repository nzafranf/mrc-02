# Error report: Qwen2.5-14B-Instruct_fs_id on official_val

Model `Qwen/Qwen2.5-14B-Instruct`, prompt `fs_id`, 7 demos, decision as generated. 1000 questions = the first 1000 of the fixed order (640 answerable, 360 unanswerable).

EM 63.0 · F1 75.1 · HasAns F1 70.8 · NoAns acc 82.8 · predicted no-answer 38.5%

## Outcomes, general to specific

```
outcome                                                 n    % all  % parent
all questions                                        1000   100.0%          
└─ correct (exact match)                              630    63.0%     63.0%
   └─ correct answer                                  332    33.2%     52.7%
   └─ correct no-answer                               298    29.8%     47.3%
└─ wrong                                              370    37.0%     37.0%
   └─ answerability error                             149    14.9%     40.3%
      └─ false no-answer (answerable, abstained)       87     8.7%     58.4%
      └─ false answer (unanswerable, gave a span)      62     6.2%     41.6%
   └─ span error (answerable, wrong span)             221    22.1%     59.7%
      └─ overlaps gold (partial credit)               189    18.9%     85.5%
         └─ boundary too long                         128    12.8%     67.7%
         └─ boundary too short                         31     3.1%     16.4%
         └─ partial overlap                            30     3.0%     15.9%
      └─ no overlap (F1 = 0)                           32     3.2%     14.5%
         └─ wrong span, same sentence                  15     1.5%     46.9%
         └─ wrong sentence                             16     1.6%     50.0%
         └─ not in the passage                          1     0.1%      3.1%
```

## Grounding of the spans the model gave

exact = copied from the passage; snapped = mapped to the closest passage span; ungrounded = not in the passage (hallucinated or reformatted).

| outcome | exact | snapped | ungrounded |
|---|---|---|---|
| correct_ans | 327 | 5 | 0 |
| false_answer_on_unanswerable | 49 | 9 | 4 |
| boundary_too_long | 105 | 21 | 2 |
| boundary_too_short | 25 | 6 | 0 |
| partial_overlap | 17 | 12 | 1 |
| wrong_span_same_sentence | 13 | 2 | 0 |
| wrong_sentence | 11 | 5 | 0 |
| not_in_passage | 0 | 0 | 1 |

## Errors by question type

| q_type | n | wrong | false_no_answer | false_answer_on_unanswerable | boundary_too_long | boundary_too_short | partial_overlap | wrong_span_same_sentence | wrong_sentence | not_in_passage |
|---|---|---|---|---|---|---|---|---|---|---|
| entity_or_def | 350 | 144 | 20 | 28 | 50 | 17 | 17 | 5 | 7 | 0 |
| location | 101 | 53 | 13 | 6 | 24 | 2 | 5 | 2 | 1 | 0 |
| number | 153 | 47 | 18 | 7 | 14 | 3 | 1 | 3 | 0 | 1 |
| other | 31 | 14 | 0 | 6 | 4 | 0 | 3 | 0 | 1 | 0 |
| person | 168 | 55 | 18 | 7 | 14 | 6 | 3 | 3 | 4 | 0 |
| time | 194 | 56 | 17 | 8 | 22 | 3 | 1 | 2 | 3 | 0 |
| yesno_like | 3 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Examples (up to 3 per error type, in the fixed order)

### false_no_answer (87)

- `qa_02dfcfe7eb6cbde4af5f` **Q:** Siapakah yang menemukan ilmu Matematika?  
  **gold:** Thales dari Miletus (kira-kira 624 sampai 546 SM) dan Pythagoras dari Samos · **pred:** (no answer) · raw `unanswerable` · noans · F1 0.00
- `qa_3b1a540ae1f4569823da` **Q:** dimanakah kantor pusat PT Kertas Kraft Aceh?  
  **gold:** desa Jamuan, kecamatan Banda Baro, kabupaten Aceh Utara, Provinsi Aceh · **pred:** (no answer) · raw `unanswerable` · noans · F1 0.00
- `qa_3ef20464d2873ad98836` **Q:** siapakah pencipta kereta listrik?  
  **gold:** Werkspoor dan Heemaf Hengelo · **pred:** (no answer) · raw `unanswerable` · noans · F1 0.00

### false_answer_on_unanswerable (62)

- `qa_f3becbe53a050c0fee28` **Q:** Di mana penggagas teori kesempurnaan media?  
  **gold:** (unanswerable) · **pred:** Richard L. Daft dan Robert H. Lengel · raw `Richard L. Daft dan Robert H. Lengel` · exact · F1 0.00
- `qa_d29d8f82e0b1ac50140c` **Q:** Berapa isteri Joko Widodo?  
  **gold:** (unanswerable) · **pred:** 1 · raw `1` · exact · F1 0.00
- `qa_d52548970d6de830f53c` **Q:** Kapan Nazi kehilangan kekuasaan di Jerman?  
  **gold:** (unanswerable) · **pred:** akhir Perang Dunia II pada tahun 1945, ketika · raw `1945, ketika akhir Perang Dunia II` · snapped · F1 0.00

### boundary_too_long (128)

- `qa_49a7badd9e1590146663` **Q:** Siapa yang menciptakan lagu God Defend New Zealand?  
  **gold:** John Joseph Woods · **pred:** Thomas Bracken (menulis puisi), John Joseph Woods (mengubah puisi menjadi lagu) · raw `Thomas Bracken (menulis puisi), John Joseph Woods (mengubah puisi menjadi lagu)` · ungrounded · F1 0.43
- `qa_de936d67f23ec497665c` **Q:** Kapan agama Buddha wujud?  
  **gold:** abad ke-2 M · **pred:** abad ke-2 M sampai sekarang dari lahirnya sang Buddha Siddharta Gautama · raw `abad ke-2 M sampai sekarang dari lahirnya sang Buddha Siddharta Gautama` · exact · F1 0.43
- `qa_4fb9bed7b74bd90b5171` **Q:** Berapa orang yang menghadiri Kongres Partai Buruh Korea ke-6?  
  **gold:** 3.062 · **pred:** 3.062 delegasi dengan hak suara dan 158 tanpa hak suara · raw `3.062 delegasi dengan hak suara dan 158 tanpa hak suara` · exact · F1 0.18

### boundary_too_short (31)

- `qa_2c68395f4b192ef4a8c2` **Q:** Apa penyebab kematian Islam Abduganiyevich Karimov?  
  **gold:** menderita stroke · **pred:** stroke · raw `stroke` · exact · F1 0.67
- `qa_35f043f582f5d93f6dc3` **Q:** berapakah luas pelabuhan Sunda kelapa?  
  **gold:** daratan 760 hektare serta luas perairan kolam 16.470 hektare · **pred:** 760 hektare serta luas perairan kolam 16.470 hektare · raw `760 hektare serta luas perairan kolam 16.470 hektare` · exact · F1 0.94
- `qa_fa6ebe2e4cc829935776` **Q:** Apakah sistem pemerintahan negara Taiwan?  
  **gold:** demokrasi dan liberalisme yang umum digunakan negara -negara barat · **pred:** demokrasi dan liberalisme · raw `demokrasi dan liberalisme` · exact · F1 0.50

### partial_overlap (30)

- `qa_86552309537310e035c7` **Q:** Mengapa Gereja Kebangunan Kalam Allah Indonesia didirikan?  
  **gold:** untuk melayani Kebaktian Kebangunan Rohani · **pred:** Melalui suatu mimpi, Tuhan memberi visi dan panggilan untuk memberitakan Injil ke Hindia Belanda (sebutan untuk Indonesia saat itu) · raw `Melalui suatu mimpi, Tuhan memberi visi dan panggilan untuk memberitakan Injil ke Hindia Belanda (sebutan untuk Indonesi` · exact · F1 0.08
- `qa_7e9d3274678778942300` **Q:** Dimanakah letak Granada ?  
  **gold:** Spanyol bagian selatan · **pred:** Spanyol. Secara geografis Granada berada pada kaki gunung Sierra Nevada, berada pada muara dari tiga sungai, yakni Beiro, Darro dan Genil. Berada pada · raw `Granada terletak di Spanyol bagian selatan, pada kaki gunung Sierra Nevada, berada pada muara dari tiga sungai, yakni Be` · snapped · F1 0.08
- `qa_f58adab8cb125863b694` **Q:** dimanakah letak pulau Sulawesi?  
  **gold:** sebelah timur Kalimantan, sebelah barat Kepulauan Maluku, dan sebelah selatan Mindanao dan Kepulauan Sulu, Filipina · **pred:** terletak di sebelah timur Kalimantan, sebelah barat Kepulauan Maluku, dan sebelah selatan Mindanao dan Kepulauan Sulu · raw `Sulawesi terletak di sebelah timur Kalimantan, sebelah barat Kepulauan Maluku, dan sebelah selatan Mindanao dan Kepulaua` · snapped · F1 0.90

### wrong_span_same_sentence (15)

- `qa_cad67483eb16cd4d85e4` **Q:** apa nama sungai terpanjang di Republik Rakyat Tiongkok?  
  **gold:** Changjiang · **pred:** Sungai Panjang · raw `Sungai Panjang` · exact · F1 0.00
- `qa_c45afd088d7005f5ffca` **Q:** Kapan Perang Salib Pertama terjadi ?  
  **gold:** 1096 · **pred:** 1096–1099 · raw `1096–1099` · exact · F1 0.00
- `qa_3d8a2614b9f548a5daba` **Q:** apakah yang dimaksud dengan Mahayana?  
  **gold:** Kendaraan Besar' · **pred:** Sang Buddha. Mahayana, yang dilahirkan di India, digunakan atas tiga pengertian utama: · raw `Sang Buddha. Mahayana, yang dilahirkan di India, digunakan atas tiga pengertian utama:` · exact · F1 0.00

### wrong_sentence (16)

- `qa_a28353b01c5abbfad015` **Q:** mengapa Paus Yohanes Paulus I menjabat sebagai Paus hanya 33 hari?  
  **gold:** meninggal dunia · **pred:** Yohanes Paulus I ditemukan tewas duduk di tempat tidurnya sebelum fajar pada tanggal 29 September 1978, hanya 33 hari ke · raw `karena ditemukan tewas duduk di tempat tidurnya sebelum fajar pada tanggal 29 September 1978, kemungkinan besar meningga` · snapped · F1 0.00
- `qa_3c921518a4155ce02dda` **Q:** Kapan Daisetz Teitaro Suzuki mulai menulis buku ?  
  **gold:** 1890 · **pred:** 1906 · raw `1906` · exact · F1 0.00
- `qa_82fae27ed7a03451861b` **Q:** Apakah isi Naskah Komentari Kitab Habakuk ?  
  **gold:** tafsir (pesyer) atau komentari dalam bahasa Ibrani · **pred:** satu ayat lengkap dari Kitab Habakuk diikuti tafsirannya, kemudian ayat berikutnya dan seterusnya · raw `satu ayat lengkap dari Kitab Habakuk diikuti tafsirannya, kemudian ayat berikutnya dan seterusnya` · exact · F1 0.00

### not_in_passage (1)

- `qa_dda952fc2819c5219388` **Q:** Berapakah panjang gelombang ultra violet?  
  **gold:** UVA (380–315nm), yang juga disebut "Gelombang Panjang" atau "blacklight"; UVB (315–280nm), yang juga disebut "Gelombang Medium" (Medium Wave); dan UVC (280-10nm), juga disebut "Gelombang Pendek" (Short Wave) · **pred:** 380–10nm · raw `380–10nm` · ungrounded · F1 0.00

