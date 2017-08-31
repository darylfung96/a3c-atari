import tensorflow as tf
from tensorflow.python.training import training_ops
from tensorflow.python.training import slot_creator

class RMSApplier():
    def __init__(self, learning_rate=0.001, decay=0.9, momentum=0.0, epsilon=1e-10, clip_value=40.0, device="/cpu:0", name="RMSApplier"):
        self._learning_rate = learning_rate
        self._decay = decay
        self._momentum = momentum
        self._epsilon = epsilon
        self._clip_value = clip_value
        self._device = device
        self._name = name

        self._dict = {}


        # we have tensors for the parameters here
        self._learning_rate_tensor = None
        self._decay_tensor = None
        self._momentum_tensor = None
        self._epsilon_tensor = None


    def _prepare(self):
        self._learning_rate_tensor = tf.convert_to_tensor(self._learning_rate, name='learning_rate')
        self._decay_tensor = tf.convert_to_tensor(self._decay, name='decay_tensor')
        self._momentum_tensor = tf.convert_to_tensor(self._momentum, name='momentum_tensor')
        self._epsilon_tensor = tf.convert_to_tensor(self._epsilon, name='epsilon_tensor')

    def _create_slots(self, var_list):
        for v in var_list:
            val = tf.constant(1.0, dtype=v.dtype, shape=v.get_shape())
            self._get_make_slots(v, val, 'rms', self._name)
            self._zero_slots(v, 'momentum', self._name)


    def _get_slot_in_dict(self, slot_name):
        name_slot = self._dict.get(slot_name, None)
        if name_slot is None:
            name_slot = {}
            self._dict[slot_name] = name_slot
        return name_slot

    # get the slot of the accum gradient for this variable
    # if doesn't exist we make them
    def _get_make_slots(self, v, val, slot_name, op_name):
        # get the slot of this accumulative gradient
        name_slot = self._get_slot_in_dict(slot_name)

        # if the variable we want to create for the accumlative gradient is not inside
        # we create it
        if v not in name_slot:
            name_slot[v] = slot_creator.create_slot(val, v, op_name)

        return name_slot[v]

    def _get_slot(self, v, name):
        name_slot = self._get_slot_in_dict(name)
        if not name_slot:
            return None
        return name_slot.get(v, None)

    # create zero slots for momentum
    def _zero_slots(self, v, slot_name, op_name):
        # get the slot for the accumulative gradient
        name_slot = self._get_slot_in_dict(slot_name)

        # if the variable is not inside name_slot we create it
        if v not in name_slot:
            name_slot[v] = slot_creator.create_zeros_slot(v, op_name)
        return name_slot[v]

    def _apply_dense(self, variable, grad):
        rms = self._get_slot(variable, 'rms')
        momentum = self._get_slot(variable, 'momentum')
        return training_ops.apply_rms_prop(variable, rms, momentum,
                                           self._learning_rate_tensor,
                                           self._decay_tensor,
                                           self._momentum_tensor,
                                           self._epsilon_tensor,
                                           grad, use_locking=False).op


    def apply_gradient(self, var_list, grad_list, name=None):
        update_grad_operations = []

        with tf.device(self._device):
            # create slots for accum_grad before applying them to global network
            with tf.control_dependencies(None):
                self._create_slots(var_list)

            with tf.name_scope(name, self._name, []) as name:
                self._prepare()
                for variable, grad in zip(var_list, grad_list):
                    clipped_grad = tf.clip_by_norm(grad, self._clip_value)
                    update_grad_operations.append(self._apply_dense(variable, clipped_grad))

                return tf.group(*update_grad_operations, name=name)
