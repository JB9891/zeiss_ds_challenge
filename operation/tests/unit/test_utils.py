import pandas as pd


def get_mock_data():
    """Creates mock data

    :return: Mock Data
    :rtype: DataFrame
    """
    mock_data = pd.DataFrame(
        {
            "fakeID": ["0000_0001", "0000_0002", "0000_0003", "0000_0004"],
            "b_specialisation_i": [0, 0, 0, 1],
            "b_specialisation_h": [1, 0, 0, 1],
            "b_specialisation_g": [0, 0, 0, 1],
            "b_specialisation_f": [0, 1, 0, 1],
            "b_specialisation_e": [0, 0, 0, 1],
            "b_specialisation_d": [1, 0, 0, 1],
            "b_specialisation_c": [0, 0, 0, 0],
            "b_specialisation_b": [0, 0, 0, 0],
            "b_specialisation_a": [0, 0, 0, 0],
            "b_specialisation_j": [0, 0, 0, 0],
            "q_OpeningDays": [2, 2, 2, 0],
            "q_OpeningHours": [10, 10, 10, 0],
            "q_2017 Average Household Size": [2, 2, 2, 2],
            "q_2017 Total Households": [10, 10, 10, 10],
            "q_2017 HHs: 5th Quintile (68.759 and above)": [
                10.5,
                10.5,
                10.5,
                10.5,
            ],
            "q_5th Quint by Total HH": [10.5, 10.5, 10.5, 10.5],
            "q_2017 Purchasing Power: Per Capita": [10.5, 10.5, 10.5, 10.5],
            "q_2017 Total Population": [10.5, 10.5, 10.5, 10.5],
            "q_2017 Pop 15+/Edu: University, Fachhochschule": [
                10.5,
                10.5,
                10.5,
                10.5,
            ],
            "q_Uni by Total Pop": [10.5, 10.5, 10.5, 10.5],
            "q_2017 Personal Care: Per Capita": [10.5, 10.5, 10.5, 10.5],
            "q_2017 Medical Products: Per Capita": [10.5, 10.5, 10.5, 10.5],
            "q_2017 Personal Effects: Per Capita": [10.5, 10.5, 10.5, 10.5],
            "b_in_kontakt_gewesen": [0, 1, 1, 1],
            "b_gekauft_gesamt": [0, 0, 1, 1],
        }
    )

    return mock_data
