# Error sample: qwen0.5b_id1_0shot_n60_first_token (qwen0.5b_id1_0shot_n60_first_token)

40 errors stratified by outcome (round robin) + 10 correct answers with the highest na_prob. Fill human_label with one of: gold_issue, multi_hop_miss, right_type_wrong_entity, wrong_type, none, other.

| # | id | outcome | question | gold | pred | q_type | human_label | notes |
|---|---|---|---|---|---|---|---|---|
| 1 | qa_04b38b92e0a507d36cd7 | false_no_answer | Siapakah Menteri Transmigrasi dan Pemukiman Proyek lahan gambut satu juta hektar? | Siswono Yudo Husodo | (no answer) | person |  |  |
| 2 | qa_180f09da0a4c0a1fd302 | wrong_sentence | Apa warna asli Angsa batu kaki biru? | coklat | angsa batu kaki biru menunjukkan warna | entity_or_def |  |  |
| 3 | qa_0674f81bcf68276feae4 | boundary_too_long | Apakah yang dimaksud dengan algoritme? | prosedur langkah-demi-langkah untuk penghitungan | komputer, algoritme adalah prosedur langkah-demi-langkah untuk penghitungan. Algoritme digunakan | entity_or_def |  |  |
| 4 | qa_074634b2eae86659178f | partial_overlap | Berapa luas Pulau Serangan? | panjang maksium 2,9 km dan lebar 1 km | Luas Pulau Serangan adalah 2,9 km persegi | number |  |  |
| 5 | qa_0d82790c6d3244970680 | boundary_too_short | siapakah orang tua Mahasthavira Vajragiri? | Ayahnya bernama The Tjai Gwan dan ibunya Liem Kiat Nio | The Tjai Gwan dan ibunya Liem Kiat Nio | person |  |  |
| 6 | qa_16a2febc8c0b6aa56cea | false_no_answer | berapakah jumlah kecamatan di Kalimantan Selatan? | 153 | (no answer) | number |  |  |
| 7 | qa_05472ac12e5fdcef4ee0 | wrong_sentence | Dimana letak Düsseldorf? | Jerman | Düsseldorf terletak di tepi Sungai Rhein | location |  |  |
| 8 | qa_14b925e5c6d46c010913 | boundary_too_long | negara manakah yang mempunyai jaringan Internet pertama kali? | Amerika Serikat | Departemen Pertahanan Amerika Serikat | location |  |  |
| 9 | qa_12a96c12a171ddb1849c | false_no_answer | Apa yang tidak ada di kereta ekonomi sebelum tahun 2010? | AC | (no answer) | entity_or_def |  |  |
| 10 | qa_0a266524498967311c7b | boundary_too_long | apakah ibukota provinsi  Kalimantan Selatan? | Banjarmasin | adalah Banjarmasin. Provinsi Kalimantan Selatan | entity_or_def |  |  |
| 11 | qa_175e4e7028a6638527bf | false_no_answer | dimanakah letak gunung Fuji di Jepang ? | perbatasan Prefektur Shizuoka dan Yamanashi, di sebelah barat Tokyo | (no answer) | location |  |  |
| 12 | qa_0a3867b0e2c4601ee016 | boundary_too_long | Berapa luas Papua? | 808.105km | Papua memiliki luas 808.105km persegi | number |  |  |
| 13 | qa_07b2bea085cd5a148a56 | false_no_answer | Sejak tanggal berapakah Jepang tidak berhak mencampuri urusan perkeretaapian di Indonesia? | 28 September 1945 | (no answer) | number |  |  |
| 14 | qa_1517da00eb83219c0d8e | boundary_too_long | Apakah rumah adat Maluku? | Rumah Baileo | adalah rumah adat Maluku dan Maluku Utara, Indonesia.[1] Rumah Baileo merupakan representasi kebudayaan Maluku | entity_or_def |  |  |
| 15 | qa_1381e371016dcc40f018 | false_no_answer | Kapan layar sentuh pada handphone pertama kali diciptakan ? | 1992 | (no answer) | time |  |  |
| 16 | qa_117ffbb8cd4ed821d9e9 | boundary_too_long | apakah nama ibukota Vietnam? | Hanoi | Ibukota Vietnam adalah Hanoi | entity_or_def |  |  |
| 17 | qa_140ea77136a742b49cea | false_no_answer | berapakah jumlah jenis fauna yang ada di Indonesia? | 147 spesies mamalia, 114 burung, 91 ikan dan 2 invertebrata | (no answer) | number |  |  |
| 18 | qa_0d3166a8ccb592c32956 | boundary_too_long | Apa nama suku asli masyarakat Banyuwangi ? | Suku Using | Suku Using adalah penduduk asli Banyuwangi | entity_or_def |  |  |
| 19 | qa_09df0b3461ac400e9981 | false_no_answer | kapankah Yesus Kristus dilahirkan? | sesudah tahun 6 SM | (no answer) | time |  |  |
| 20 | qa_14c02949ccea2ed9dabd | false_no_answer | Kapan Kinetoskop diciptakan ? | 1889 / 1889 dan 1892 | (no answer) | time |  |  |
| 21 | qa_0e2de9da8e47e8e3bce2 | false_no_answer | Bahasa apa yang digunakan oleh Wayang Golek Menak? | Jawa | (no answer) | entity_or_def |  |  |
| 22 | qa_0406f9996f7eb6dcd774 | false_no_answer | Siapa presiden pertama Nauru? | Hammer DeRoburt | (no answer) | person |  |  |
| 23 | qa_00e198cdbe7a6db9fff7 | false_no_answer | Kapan Yan Xishan diangkat menjadi panglima perang Tiongkok ? | 1911 | (no answer) | time |  |  |
| 24 | qa_120d3e2aeafdd83b1379 | false_no_answer | Kapan Minecraft pertama kali diciptakan? | 17 Mei 2011 | (no answer) | time |  |  |
| 25 | qa_1736e82de25b5f57abdc | false_no_answer | Pada suhu berapa air mendidih ? | 0°C | (no answer) | number |  |  |
| 26 | qa_05aed54e82bdb9edebf5 | false_no_answer | Apakah ibukota Papua Nugini? | Port Moresby | (no answer) | entity_or_def |  |  |
| 27 | qa_0349eeca8eb4cae07506 | false_no_answer | kapankah papua nugini melepaskan diri dari indonesia? | 1 Juli 1971 | (no answer) | time |  |  |
| 28 | qa_08d2b4d14fb91aadc8be | false_no_answer | Kapan Vladimir Sviatoslavich pindah agama menjadi Kristen ? | 988 | (no answer) | time |  |  |
| 29 | qa_01364a9092f96eb6e2be | false_no_answer | Apakah kota terbesar di Iran? | Teheran | (no answer) | entity_or_def |  |  |
| 30 | qa_15fcf4146498cca19dcf | false_no_answer | Siapakah yang tidak sedikit memberikan kontribusi bagi berdirinya Dinasti Tang? | Li Jiancheng | (no answer) | person |  |  |
| 31 | qa_006703b670a2de86a02f | false_no_answer | kapankah Uni Eropa didirikan? | 1993 | (no answer) | time |  |  |
| 32 | qa_17e01bfa1eeedaf3a2be | false_no_answer | Kapan ilmu astronomi diperkenalkan ? | abad ke-3 SM | (no answer) | time |  |  |
| 33 | qa_032e963768b50c42794c | false_no_answer | Apakah ibu kota Tanzania? | Dar-es-Salam | (no answer) | entity_or_def |  |  |
| 34 | qa_0bf9cafef11db24ae688 | correct_low_conf | Siapa yang menemukan Tabel periodik? | Dmitri Mendeleev | Dmitri Mendeleev | person |  |  |
| 35 | qa_0986961735e0c79ff572 | correct_low_conf | Siapakah nama pemimpin dalam gereja ? | Paus | Paus | person |  |  |
| 36 | qa_11f43b983ed2d07230d3 | correct_low_conf | apakah nama ibukota Israel? | Tel Aviv | Tel Aviv | entity_or_def |  |  |
