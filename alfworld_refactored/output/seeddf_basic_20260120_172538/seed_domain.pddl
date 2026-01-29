(define (domain exploration)
  (:requirements :strips :typing :negative-preconditions)
  (:types
    receptacle object - thing
    microwaveReceptacle sinkbasinReceptacle fridgeReceptacle - receptacle
    sharpObject - object
  )
  (:predicates
    (at ?r - receptacle)
    (opened ?r - receptacle)
    (in ?o - object ?r - receptacle)
    (holding ?o - object)
    (handempty)
    (heated ?o - object)
    (cleaned ?o - object)
    (cooled ?o - object)
    (sliced ?o - object)
    (on ?o - object)
  )
  (:action GOTOLOCATION
    :parameters (?from - receptacle ?to - receptacle)
    :precondition (at ?from)
    :effect (and (not (at ?from)) (at ?to))
  )
  (:action OPENOBJECT
    :parameters (?r - receptacle)
    :precondition (and (at ?r) (not (opened ?r)))
    :effect (opened ?r)
  )
  (:action CLOSEOBJECT
    :parameters (?r - receptacle)
    :precondition (and (at ?r) (opened ?r))
    :effect (not (opened ?r))
  )
  (:action PICKUPOBJECT
    :parameters (?o - object ?r - receptacle)
    :precondition (and (at ?r) (in ?o ?r) (handempty))
    :effect (and (holding ?o) (not (in ?o ?r)) (not (handempty)))
  )
  (:action PUTOBJECT
    :parameters (?o - object ?r - receptacle)
    :precondition (and (at ?r) (holding ?o))
    :effect (and (in ?o ?r) (handempty) (not (holding ?o)))
  )
  (:action USEOBJECT
    :parameters (?o - object)
    :precondition (holding ?o)
    :effect (on ?o)
  )
  (:action HEATOBJECT
    :parameters (?o - object ?r - microwaveReceptacle)
    :precondition (and (at ?r) (holding ?o))
    :effect (heated ?o)
  )
  (:action CLEANOBJECT
    :parameters (?o - object ?r - sinkbasinReceptacle)
    :precondition (and (at ?r) (holding ?o))
    :effect (cleaned ?o)
  )
  (:action COOLOBJECT
    :parameters (?o - object ?r - fridgeReceptacle)
    :precondition (and (at ?r) (holding ?o))
    :effect (cooled ?o)
  )
  (:action SLICEOBJECT
    :parameters (?r - receptacle ?co - object ?sharp_o - sharpObject)
    :precondition (and (at ?r) (in ?co ?r) (holding ?sharp_o))
    :effect (sliced ?co)
  )
)