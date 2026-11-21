# Error sample: qwen0.5b_id1_3shot_fixed_n100 (qwen0.5b_id1_3shot_fixed_n100)

40 errors stratified by outcome (round robin) + 10 correct answers with the highest na_prob. Fill human_label with one of: gold_issue, multi_hop_miss, right_type_wrong_entity, wrong_type, none, other.

| # | id | outcome | question | gold | pred | q_type | human_label | notes |
|---|---|---|---|---|---|---|---|---|
| 1 | qa_1aaf53e8bdeae8febcd3 | false_no_answer | Siapa penemu Aljabar? | Al-Khwārizmī | (no answer) | person |  |  |
| 2 | qa_04b38b92e0a507d36cd7 | wrong_sentence | Siapakah Menteri Transmigrasi dan Pemukiman Proyek lahan gambut satu juta hektar? | Siswono Yudo Husodo | satu juta hektar lahan gambut dan rawa untuk penanaman padi.[3] Proyek | person |  |  |
| 3 | qa_06c9c5c2f0f891b0e163 | false_answer_on_unanswerable | apakah yang di maksud dengan membran yang berwarna merah? | (no answer) | selaput, kulit tipis, atau lembaran bahan tipis, yang berfungsi sebagai pem | entity_or_def |  |  |
| 4 | qa_14b925e5c6d46c010913 | boundary_too_long | negara manakah yang mempunyai jaringan Internet pertama kali? | Amerika Serikat | Departemen Pertahanan Amerika Serikat | location |  |  |
| 5 | qa_1e53abdcb86f5d50175c | partial_overlap | Apakah yang dimaksud Gelombang gravitasi? | riak dalam lengkung ruang-waktu yang bergerak dalam bentuk gelombang menjauhi sumbernya | riak dalam lengkung ruang-waktu yang bergerak dalam bentuk gelombang menjauhi s | entity_or_def |  |  |
| 6 | qa_0a3867b0e2c4601ee016 | false_no_answer | Berapa luas Papua? | 808.105km | (no answer) | number |  |  |
| 7 | qa_117ffbb8cd4ed821d9e9 | boundary_too_long | apakah nama ibukota Vietnam? | Hanoi | Ibukota Vietnam adalah Hanoi | entity_or_def |  |  |
| 8 | qa_1517da00eb83219c0d8e | partial_overlap | Apakah rumah adat Maluku? | Rumah Baileo | rumah adat Maluku | entity_or_def |  |  |
| 9 | qa_032e963768b50c42794c | false_no_answer | Apakah ibu kota Tanzania? | Dar-es-Salam | (no answer) | entity_or_def |  |  |
| 10 | qa_0a266524498967311c7b | boundary_too_long | apakah ibukota provinsi  Kalimantan Selatan? | Banjarmasin | Ibu kotanya adalah Banjarmasin | entity_or_def |  |  |
| 11 | qa_2601cd3cac7fadee10ec | partial_overlap | apakah bilangan riil? | bilangan yang bisa dituliskan dalam bentuk desimal | Bilangan riil | entity_or_def |  |  |
| 12 | qa_00e198cdbe7a6db9fff7 | false_no_answer | Kapan Yan Xishan diangkat menjadi panglima perang Tiongkok ? | 1911 | (no answer) | time |  |  |
| 13 | qa_0d3166a8ccb592c32956 | partial_overlap | Apa nama suku asli masyarakat Banyuwangi ? | Suku Using | Suku Osing | entity_or_def |  |  |
| 14 | qa_1e61ddf3a44c0ea91ba9 | false_no_answer | Kapan Red Hot Chili Peppers dibentuk? | 1983 | (no answer) | time |  |  |
| 15 | qa_180f09da0a4c0a1fd302 | false_no_answer | Apa warna asli Angsa batu kaki biru? | coklat | (no answer) | entity_or_def |  |  |
| 16 | qa_214ca9b6b63b3090911d | false_no_answer | Apa penyebab utama Penyakit pulmonari obstruktif kronis? | Merokok tembakau | (no answer) | entity_or_def |  |  |
| 17 | qa_05472ac12e5fdcef4ee0 | false_no_answer | Dimana letak Düsseldorf? | Jerman | (no answer) | location |  |  |
| 18 | qa_11f43b983ed2d07230d3 | false_no_answer | apakah nama ibukota Israel? | Tel Aviv | (no answer) | entity_or_def |  |  |
| 19 | qa_0349eeca8eb4cae07506 | false_no_answer | kapankah papua nugini melepaskan diri dari indonesia? | 1 Juli 1971 | (no answer) | time |  |  |
| 20 | qa_0d82790c6d3244970680 | false_no_answer | siapakah orang tua Mahasthavira Vajragiri? | Ayahnya bernama The Tjai Gwan dan ibunya Liem Kiat Nio | (no answer) | person |  |  |
| 21 | qa_20564c45de2060da8fb1 | false_no_answer | Kapan Yesus lahir ? | 6 SM | (no answer) | time |  |  |
| 22 | qa_1381e371016dcc40f018 | false_no_answer | Kapan layar sentuh pada handphone pertama kali diciptakan ? | 1992 | (no answer) | time |  |  |
| 23 | qa_15fcf4146498cca19dcf | false_no_answer | Siapakah yang tidak sedikit memberikan kontribusi bagi berdirinya Dinasti Tang? | Li Jiancheng | (no answer) | person |  |  |
| 24 | qa_215981b00544ba9a3c1c | false_no_answer | Apakah kesenian tradisional khas Gunungkidul yang mencerminkan bahwa kehidupan manusia tidak jauh dari alam? | Rinding Gumbeng | (no answer) | entity_or_def |  |  |
| 25 | qa_0406f9996f7eb6dcd774 | false_no_answer | Siapa presiden pertama Nauru? | Hammer DeRoburt | (no answer) | person |  |  |
| 26 | qa_233a408f80fd80b4c302 | false_no_answer | Apakah nama ibukota Jepang? | Tokyo | (no answer) | entity_or_def |  |  |
| 27 | qa_205eb0af618660bf8d4e | false_no_answer | Apakah nama ibukota Kalimantan Timur? | Samarinda | (no answer) | entity_or_def |  |  |
| 28 | qa_1f4e980b4aba401ebe8e | false_no_answer | berapakah luas Papua? | 808.105km persegi | (no answer) | number |  |  |
| 29 | qa_1a05439cf9633a04e7eb | false_no_answer | apakah nama ibukota Polandia ? | Warsawa | (no answer) | entity_or_def |  |  |
| 30 | qa_1e4ba2c2a06dde242174 | false_no_answer | Kapan kekaisaran Romawi Timur runtuh? | 29 Mei 1453 | (no answer) | time |  |  |
| 31 | qa_244f531e28dd211e2fb1 | false_no_answer | Siapakah tokoh antagonis dalam sinetron Mutiara Hati ? | Marchia Caroline Pohan | (no answer) | person |  |  |
| 32 | qa_1e0375bb9d9e7cb9488b | false_no_answer | Apa ibukota Israel ? | Tel Aviv | (no answer) | entity_or_def |  |  |
| 33 | qa_09df0b3461ac400e9981 | false_no_answer | kapankah Yesus Kristus dilahirkan? | sesudah tahun 6 SM | (no answer) | time |  |  |
| 34 | qa_20ef37ed8d030f583cee | false_no_answer | berapakah luas Kanada? | 9.970.610 kilometer persegi | (no answer) | number |  |  |
| 35 | qa_074634b2eae86659178f | false_no_answer | Berapa luas Pulau Serangan? | panjang maksium 2,9 km dan lebar 1 km | (no answer) | number |  |  |
| 36 | qa_0986961735e0c79ff572 | false_no_answer | Siapakah nama pemimpin dalam gereja ? | Paus | (no answer) | person |  |  |
| 37 | qa_0e2de9da8e47e8e3bce2 | false_no_answer | Bahasa apa yang digunakan oleh Wayang Golek Menak? | Jawa | (no answer) | entity_or_def |  |  |
| 38 | qa_2548a969bbcdbeb6bc0d | false_no_answer | Kapan telegram pertama ditemukan? | 1837 | (no answer) | time |  |  |
| 39 | qa_14c02949ccea2ed9dabd | false_no_answer | Kapan Kinetoskop diciptakan ? | 1889 / 1889 dan 1892 | (no answer) | time |  |  |
| 40 | qa_17e01bfa1eeedaf3a2be | false_no_answer | Kapan ilmu astronomi diperkenalkan ? | abad ke-3 SM | (no answer) | time |  |  |
| 41 | qa_24041b11d835d85aa0ca | correct_low_conf | Apa nama bandara pertama di Indonesia? | Bandar Udara Kemayoran | Bandar Udara Kemayoran | entity_or_def |  |  |
| 42 | qa_1cafbf04e5059f20d957 | correct_low_conf | siapakah karakter uatama anime One Piece? | Monkey D. Luffy | Monkey D. Luffy | person |  |  |
| 43 | qa_0bf9cafef11db24ae688 | correct_low_conf | Siapa yang menemukan Tabel periodik? | Dmitri Mendeleev | Dmitri Mendeleev | person |  |  |
| 44 | qa_0674f81bcf68276feae4 | correct_low_conf | Apakah yang dimaksud dengan algoritme? | prosedur langkah-demi-langkah untuk penghitungan | prosedur langkah-demi-langkah untuk penghitungan | entity_or_def |  |  |
