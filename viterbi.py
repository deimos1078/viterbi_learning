def viterbi(observations: tuple,
            states: set,
            start_probabilities: dict,
            transitions_probabilities: dict,
            emissions_probabilities: dict) -> list:
    """
    :param observations: sequence of observations
    :param states: set of all possible states
    :param start_probabilities: dictionary of starting probabilities for each state
    :param transitions_probabilities: dictionary of transition probabilities for each pair of states
    :param emissions_probabilities: dictionary of emission probabilities for each state x emission pair
    :return: list of the most probable sequence of states
    """
    viterbi_path = [{}]
    for current_state in states:
        # Set the first column of the dynamic programming table to the starting probabilities
        viterbi_path[0][current_state] = {
            "probability": start_probabilities[current_state] * emissions_probabilities[current_state][observations[0]],
            "previous": None
        }
    # For each column of the dynamic programming table and for each possible state, look at each possible previous state
    # Calculate the probability of the transition to this state occurring under the emission observed
    # Then pick the most probable transition
    for observation_number in range(1, len(observations)):
        max_transition_probability = 0.0
        best_from_state = None
        viterbi_path.append({})
        for to_state in states:
            for from_state in states:
                # probability of being at the from_state on previous observation number
                previous_state_probability = viterbi_path[observation_number - 1][from_state]["probability"]
                # probability of transition from from_state to to_state
                transition_probability = transitions_probabilities[from_state][to_state]
                # probability of observing this emission while at to_state on the current observation number
                emission_probability = emissions_probabilities[to_state][observations[observation_number]]

                # probability that we arrived at this state, taking this path while witnessing this emission
                current_state_probability = previous_state_probability * transition_probability * emission_probability

                # if this is the highest probability we've found so far, save the path
                if current_state_probability > max_transition_probability:
                    max_transition_probability = current_state_probability
                    best_from_state = from_state

            viterbi_path[observation_number][to_state] = {"probability": max_transition_probability,
                                                          "previous": best_from_state}

    most_probable_state_sequence = []
    max_probability = 0.0
    best_state = None
    # Get most probable state and its backtrack
    for current_state, dp_cell in viterbi_path[-1].items():
        if dp_cell["probability"] > max_probability:
            max_probability = dp_cell["probability"]
            best_state = current_state
    most_probable_state_sequence.append(best_state)
    previous = best_state

    # Backtrack till the first observation
    for observation_number in range(len(viterbi_path) - 2, -1, -1):
        most_probable_state_sequence.insert(0, viterbi_path[observation_number + 1][previous]["previous"])
        previous = viterbi_path[observation_number + 1][previous]["previous"]

    return most_probable_state_sequence


if __name__ == "__main__":
    data = {
        'observations': tuple(emission for emission in '01000000100000000000001000000010001010010001001001'),
        'states': {"M", "P"},
        'start_probabilities': {'M': 0.5, 'P': 0.5},
        'transitions_probabilities': {'P': {'P': 0.95, 'M': 0.05}, 'M': {'P': 0.034482758620689655, 'M': 0.9655172413793104}},
        'emissions_probabilities': {'P': {'1': 0.3333333333333333, '0': 0.6666666666666666}, 'M': {'1': 0.10344827586206896, '0': 0.896551724137931}}
    }

    optimal_path = viterbi(data['observations'],
                           data['states'],
                           data['start_probabilities'],
                           data['transitions_probabilities'],
                           data['emissions_probabilities'])

    print("".join(data['observations']))
    print("".join(optimal_path))

