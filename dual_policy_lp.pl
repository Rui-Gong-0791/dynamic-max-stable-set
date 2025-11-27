application "Polytope";
use strict;
use warnings;
use subs qw(max_interior_simplices_impl Scalar);  # Fix bareword errors
use List::Util qw(any);

# ========================
# Helper Functions
# ========================
sub generate_subsets {
    my ($set) = @_;
    my @subsets;
    for my $mask (0..(1 << @$set)-1) {
        my @subset = map { $set->[$_] } grep { $mask & (1 << $_) } 0..$#$set;
        push @subsets, \@subset;
    }
    return @subsets;
}

sub state_to_key {
    my ($S) = @_;
    return join(',', sort @$S);
}

# ========================
# Generate States
# ========================
my @vertices = (1, 2, 3);
my %states;

for my $t (1..4) {
    if ($t == 1) {
        $states{1} = [ [@vertices] ];
    } else {
        my %new_states;
        foreach my $prev_S (@{ $states{$t-1} }) {
            # Do nothing transition
            $new_states{state_to_key([])} = 1;
            
            # Pick vertex transitions
            foreach my $i (@$prev_S) {
                my $remaining = [grep { $_ != $i } @$prev_S];
                foreach my $S_new (generate_subsets($remaining)) {
                    $new_states{state_to_key($S_new)} = 1;
                }
            }
        }
        $states{$t} = [ map { [split ',', $_] } keys %new_states ];
    }
}

# ========================
# Assign Variable Indices
# ========================
my %var_index;
my $var_count = 0;

for my $t (1..4) {
    foreach my $S (@{ $states{$t} }) {
        my $key = state_to_key($S);
        
        if ($t < 4) {
            # Do nothing action
            $var_index{"$key,$t,do"} = $var_count++;
            
            # Pick actions
            if (@$S > 0) {
                foreach my $i (@$S) {
                    $var_index{"$key,$t,pick$i"} = $var_count++;
                }
            }
        } else {
            # Terminal state only allows "do nothing"
            $var_index{"$key,4,do"} = $var_count++;
        }
    }
}

# ========================
# Build Constraints
# ========================
my @equations;
my @inequalities;

# Flow conservation constraints
for my $t (1..4) {
    foreach my $S (@{ $states{$t} }) {
        my $key = state_to_key($S);
        my @equation = (0) x $var_count;
        
        # Inflow from previous states
        if ($t > 1) {
            foreach my $prev_S (@{ $states{$t-1} }) {
                my $prev_key = state_to_key($prev_S);
                
                # Do nothing transition
                if ($key eq state_to_key([])) {
                    my $var = $var_index{"$prev_key,".($t-1).",do"};
                    $equation[$var] += 1 if defined $var;
                }
                
                # Pick transitions
                foreach my $i (@$prev_S) {
                    my $var = $var_index{"$prev_key,".($t-1).",pick$i"};
                    next unless defined $var;
                    
                    # Calculate transition probability
                    my @candidate = grep { $_ != $i } @$prev_S;
                    my %S_lookup = map { $_ => 1 } @$S;
                    my @missing = grep { !exists $S_lookup{$_} } @candidate;
                    my @present = grep { exists $S_lookup{$_} } @candidate;
                    
                    my $prob = (0.3 ** @missing) * (0.7 ** @present);
                    $equation[$var] += $prob;
                }
            }
        } else {
            # Initial state inflow = 1
            $equation[0] = 1 if $key eq state_to_key(\@vertices);
        }
        
        # Outflow to current actions
        if ($t < 4) {
            # Do nothing outflow
            my $var_do = $var_index{"$key,$t,do"};
            $equation[$var_do] -= 1 if defined $var_do;
            
            # Pick outflows
            if (@$S > 0) {
                foreach my $i (@$S) {
                    my $var = $var_index{"$key,$t,pick$i"};
                    $equation[$var] -= 1 if defined $var;
                }
            }
        }
        
        push @equations, \@equation;
    }
}

# Non-negativity constraints
for my $i (0..$var_count-1) {
    my @ineq = (0) x $var_count;
    $ineq[$i] = 1;
    push @inequalities, \@ineq;
}

# ========================
# Create Polytope
# ========================
my $lp = new Polytope(
    EQUATIONS => \@equations,
    INEQUALITIES => \@inequalities
);

print "Total variables: ", $lp->AMBIENT_DIM, "\n";
print "Total constraints: ", scalar(@{$lp->FACETS}), "\n";
print "Extreme points:\n", $lp->VERTICES;