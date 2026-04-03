import msprime, tskit, pyslim
import sys

# Get input and output files from command line arguments
if len(sys.argv) != 3:
    print("Usage: python LASLiM_neutralmoverlay.py <input_file> <output_file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

print(f"Loading tree sequence from: {input_file}")
ts = tskit.load(input_file)

def is_fully_coalesced(ts):
    for tree in ts.trees():
        if tree.num_roots > 1:
            return False
    return True

if is_fully_coalesced(ts):
    print("The tree sequence is coalesced.")
else:
    print("The tree sequence is not coalesced. Recapitation is required.")

if not is_fully_coalesced(ts):
    demography = msprime.Demography()
    demography.add_population(name="pop_0", initial_size=2000)  # unused, but named??
    demography.add_population(name="p1", initial_size=2000)

    recapitated_ts = pyslim.recapitate(ts, recombination_rate=1e-8, demography=demography)
    print("recapitation complete.")
else:
    recapitated_ts = ts
    
#print("checking recapitation...")
for t in recapitated_ts.trees():
         assert t.num_roots == 1, ("not coalesced! on segment {} to {}".
              format(t.interval[0], t.interval[1]))

        
recapitated_ts = recapitated_ts.simplify()

def print_mutation_details(ts):
    for site in ts.sites():
        for mutation in site.mutations:
            print(f"Site ID {site.id}, Position {site.position}, Derived State: {mutation.derived_state}, "
                  f"Node ID: {mutation.node}, Parent ID: {mutation.parent}")

# Print mutation details
# print("Existing mutations in the tree sequence:")
# print_mutation_details(recapitated_ts)

mutated = msprime.sim_mutations(ts, rate=1e-8, random_seed=1, keep=True)
print(f"Saving neutral overlay to: {output_file}")
mutated.dump(output_file)