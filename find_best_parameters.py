from viterbi import viterbi


# Input:
# list of observed emissions
# list of inner states

# Output:
# prediction of hidden parameters -> state to state transmissions and emission while in state probabilities
def find_best_parameters(observed_emissions: tuple,
                         inner_states: tuple,
                         max_iterations=10000,
                         initial_probabilities=None) -> tuple[dict, dict, dict]:
    # Create an initial guess of probabilities from observed emissions and information about hidden states

    # Get the structure of the hidden markov model
    iteration_number = 0
    start_probabilities, transition_probabilities, emission_probabilities = {}, {}, {}
    possible_states = set(inner_states)
    print("Initial inner states: " + "".join(inner_states))

    while iteration_number < max_iterations:
        print(f"#{iteration_number}: ")
        prev_inner_states = inner_states

        # Calculate the guess of parameters
        if iteration_number == 0 and initial_probabilities is not None:
            start_probabilities = initial_probabilities[0]
            transition_probabilities = initial_probabilities[1]
            emission_probabilities = initial_probabilities[2]
        else:
            emission_probabilities, start_probabilities, transition_probabilities = calculate_parameters(
                inner_states, observed_emissions, possible_states)

        print("Probabilities estimates: ")
        print(start_probabilities, transition_probabilities, emission_probabilities)

        # Use viterbi to guess the sequence of inner states based on those calculated parameters
        inner_states = viterbi(observed_emissions, possible_states, start_probabilities,
                               transition_probabilities, emission_probabilities)

        print("Most probable path: " + "".join(inner_states))

        iteration_number += 1

        if inner_states == prev_inner_states:
            break

    print(f"Total viterbi iterations: {iteration_number}")
    return start_probabilities, transition_probabilities, emission_probabilities


def calculate_parameters(inner_states, observed_emissions, possible_states):
    possible_emissions = tuple(set(observed_emissions))

    # The starting probabilities are evenly distributed
    start_probabilities = {state: 1 / len(possible_states) for state in possible_states}

    # Initialize
    transition_probabilities = {state: {state: 0.0 for state in possible_states} for state in possible_states}
    emission_probabilities = {state: {emission: 0.0 for emission in possible_emissions} for state in possible_states}
    stats = {
        "transition_counts": {state: {state: 0 for state in possible_states} for state in possible_states},
        "emission_counts": {state: {emission: 0 for emission in possible_emissions} for state in possible_states}
    }

    # Calculate total counts of emissions and transitions
    for i in range(len(observed_emissions)):
        stats["emission_counts"][inner_states[i]][observed_emissions[i]] = (
                stats["emission_counts"][inner_states[i]][observed_emissions[i]] + 1)

        if i != len(observed_emissions) - 1:
            stats["transition_counts"][inner_states[i]][inner_states[i + 1]] = (
                    stats["transition_counts"][inner_states[i]][inner_states[i + 1]] + 1)

    for from_state in possible_states:
        # Get total transitions from state to all states
        from_state_total_counts = 0
        for to_state in possible_states:
            from_state_total_counts += stats["transition_counts"][from_state][to_state]

        # Probability of transition from state A to state B is the amount of transitions A -> B divided
        # by the total amount of transitions from state A
        for to_state in possible_states:
            if from_state_total_counts == 0:
                transition_probabilities[from_state][to_state] = 0.0
                continue
            transition_probabilities[from_state][to_state] = (
                    stats["transition_counts"][from_state][to_state] / from_state_total_counts)

    for from_state in possible_states:
        # Get total emissions of state
        from_state_total_counts = 0
        for emission in possible_emissions:
            from_state_total_counts += stats["emission_counts"][from_state][emission]

        # Probability of emission of outer state b from inner state A is the amount of emissions A -> b
        # divided by the total amount of emissions from inner state A
        for emission in possible_emissions:
            if from_state_total_counts == 0:
                emission_probabilities[from_state][emission] = 0.0
                continue
            emission_probabilities[from_state][emission] = (
                    stats["emission_counts"][from_state][emission] / from_state_total_counts
            )

    return emission_probabilities, start_probabilities, transition_probabilities


if __name__ == "__main__":
    start_probabilities, transition_probabilities, emission_probabilities = find_best_parameters(
        tuple(emission for emission in '01000000100000000000001000000010001010010001001001'),
        tuple(state for state in 'PPPMMMMMMMMMMMMMMMMMMMMMMMMMMPPPPPPPPPPPPPPPPPPPPP'))

    print(start_probabilities)
    print(transition_probabilities)
    print(emission_probabilities)
