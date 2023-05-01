import math
import paxos


def shannon_entropy(string):
  """Calculates the Shannon entropy of a string"""
  # create hash table of character frequencies
  freqency = {}
  for char in string:
    if char in freqency:
        freqency[char] += 1
    else:
        freqency[char] = 1
  # calculate shannon entropy
  entropy = 0
  for count in freqency.values():
      probability = count / len(string)
      entropy -= probability * math.log(probability, 2)
  return entropy

# distinguish between real and fake (Paxos algorithm)
class PaxosNode:
  def __init__(self, node_id):
    self.node_id = node_id
    self.proposal_number = 0
    self.accepted_number = 0
    self.accepted_value = None
    self.acceptors = []
    self.proposer = []
  
  def prepare(self, proposal_number):
    self.proposal_number = max(self.proposal_number, proposal_number)
    promises = [acceptor.promise(proposal_number) for acceptor in self.acceptors]
    return promises
    
  def propose(self, value):
    self.proposal_number += 1
    promise_count = 0
    for proposer in self.proposers:
      promise_count += proposer.prepare(self.proposal_number)
    if promise_count >= len(self.acceptors) / 2:
      accepted_count = 0
      for proposer in self.proposers:
        if proposer.propose(value, self.proposal_number):
          accepted_count += 1
      if accepted_count >= len(self.acceptors) / 2:
        self.accepted_number = value
        return True
    return False

class PaxosAcceptor:
  def __init__(self, node_id):
    self.node_id = node_id
    self.promised_number = 0
    self.accepted_number = None
    self.accepted_value = None
  
  def promise(self, proposal_number):
    if proposal_number > self.promised_number:
      self.promised_number = proposal_number
      return (self.accepted_number, self.accepted_value)
    else:
      return None
  
  def accept(self, proposal_number, value):
    if proposal_number >= self.promised_number:
      self.promised_number = proposal_number
      self.accepted_number = proposal_number
      self.accepted_value = value
      return True
    else:
      return False

def max_entropy(strings):
  '''returns string with highest entropy'''

  max_entropy = -1
  max_string = None
  for string in strings:
    entropy = shannon_entropy(string)
    if entropy > max_entropy:
      max_entropy = entropy
      max_string = string
  return max_string

