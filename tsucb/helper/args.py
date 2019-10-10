
#  Copyright (c) 2019 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.


import argparse
import types
from pprint import pprint

from typeguard import check_type
from tsucb.helper import util


# ========================================================================= #
# Inspired as a combination of the following libraries:                     #
#     - argparse        (how variables are cast)                            #
#     - python-fire     (how classes are converted to commands)             #
#     - attrs           (how classes are annotated and variables defined)   #
# ========================================================================= #


_NONE = object()


# acts like a lazy property, but intended for use with the Args class
class computed(object):
    def __init__(self, func):
        self.__name__ = func.__name__
        self.__module__ = func.__module__
        self.__doc__ = func.__doc__
        self._func = func
        self._value = _NONE

    def __get__(self, obj, type=None):
        if self._value is _NONE:
            self._value = self._func(obj)
        return self._value

class field(object):
    def __init__(self, default=_NONE, cast=None, choices=None, required=False):
        self._default = default
        assert default is not None, 'None is not supported as a default value. TODO: parsing from the command line requires work'
        assert cast is None or callable(cast), 'Cast must be callable'
        if required:
            assert default is _NONE, 'Required cannot have a default value'
        self._cast = cast
        self._choices = set(choices) if (choices is not None) else None
        # extra info
        self._required = required
        # Set Externally In _AttrMeta
        self._type: type = _NONE
        self._name: str = _NONE
        # TODO: assert no choices, and no casts when type is bool

    def get(self, value=_NONE):
        assert (self._type is not _NONE) and (self._name is not _NONE), 'This field is not part of an Attr class.'
        # DEFAULT
        if value is _NONE:
            assert self._default is not _NONE, f'No default value specified for field: {self._name}'
            value = self._default
        # TYPE
        try:
            value = self._type(value)
        except:
            pass
        check_type(self._name, value, self._type)
        # CAST
        if self._cast:
            value = self._cast(value)
        # CHOICE
        if self._choices:
            assert value in self._choices, f'{self._name}: Value is not one of the allowed choices: {self._choices}'
        return value


class _AttrMeta(type):
    def __init__(cls, name, bases, dct):
        # INIT
        super().__init__(name, bases, dct)

        # RETURN EARLY
        if f'_{name}__is_base_attr_class' in dct:
            del dct[f'_{name}__is_base_attr_class']
            return

        # FIELDS
        cls_fields = {k: v for k, v in dct.items() if not k.startswith('__')}
        cls_fields = {k: v for k, v in cls_fields.items() if not isinstance(v, (types.FunctionType, staticmethod, classmethod, property))}

        # ANNOTATIONS
        cls_annotes = {k: v for k, v in dct.get('__annotations__', {}).items() if not k.startswith('__')}
        assert len(cls_annotes) > 0, f'Class "{name}" must have at least one field with a type annotation.'

        # CHECK ALL ANNOTATIONS HAVE DEFAULT VALUES
        for k in cls_annotes:
            if k not in cls_fields:
                raise ValueError(f'Annotated field "{k}" has no default value')

        # INITIALISE FIELDS
        fields = {}
        for k, t in cls_annotes.items():
            fields[k] = cls_fields[k] if isinstance(cls_fields[k], field) else field(default=cls_fields[k])
            fields[k]._type = t
            fields[k]._name = k

        # ADD TO TYPE
        cls._fields_ = fields
        cls._fields_public_ = {k: v for k, v in fields.items() if not k.startswith('_')}
        cls._fields_required_ = {k: v for k, v in cls._fields_public_.items() if v._required}
        cls._fields_computed = {k: v for k, v in cls_fields.items() if isinstance(v, computed)}

        # properties = {k for k, v in cls._fields_computed.items() if isinstance(v, property)}
        # if properties:
        #     props_str = ', '.join(f'"{p}"' for p in properties)
        #     print(f'WARNING: "{name}" has @property decorators: {props_str} did you mean @computed?')

class Args(object, metaclass=_AttrMeta):
    __is_base_attr_class = True

    def __init__(self, **kwargs):
        # CALL SUPER
        super().__init__()
        # CHECK REQUIRED:
        for k in self._fields_required_:
            assert k in kwargs, f'field "{k}" is required'
        # CHECK KWARGS KEYS
        for k in kwargs:
            if k in self._fields_:
                if k not in self._fields_public_:
                    raise KeyError(f'field "{k}" is not a public, cannot set from arguments.')
            else:
                raise KeyError(f'field "{k}" is unknown')
        # ASSIGN VALUES
        for k, field in self._fields_.items():
            value = field.get(kwargs[k]) if (k in kwargs) else field.get()
            setattr(self, k, value)

    def as_dict(self, only_used=False, only_changed=False) -> dict:
        # GET ALL ITEMS:
        items = {k: getattr(self, k) for k in self._fields_public_}
        if only_changed:
            changed_items = {}
            for k, v in items.items():
                field = self._fields_[k]
                if field._required or v != field._default:
                    changed_items[k] = v
            return changed_items
        # GET USED ONLY:
        if only_used:
            items = self._filter_return_used_args(items)
        # RETURN
        return items

    def get_dict_computed(self):
        return {k: getattr(self, k) for k in self._fields_computed}

    def print_dict_computed(self):
        util.print_separator('COMPUTED VALUES:')
        pprint(self.get_dict_computed(), width=100)
        print()

    def print_args(self, only_used=True, only_changed=False):
        util.print_separator(f'ARGUMENTS: (used={only_used}, defaults={not only_changed})')
        pprint(self.as_dict(only_used=only_used, only_changed=only_changed), width=100)
        print()

    def _filter_return_used_args(self, items) -> dict:
        raise NotImplementedError('')

    def as_tuple(self) -> tuple:
        return tuple(getattr(self, k) for k in self._fields_public_)

    def as_args(self, only_used=False, only_changed=False) -> str:
        strings = []
        for k, v in self.as_dict(only_used=only_used, only_changed=only_changed).items():
            flag = k.replace('_', '-')
            if self._fields_[k]._type is bool:
                if self._fields_[k]._default == v:
                    continue
                strings.append(f'--{flag}')
            else:
                v = str(v)
                assert '"' not in v, f'Invalid character \' " \' encountered in string representation of field "{k}"'
                strings.append(f'--{flag}="{v}"')
        return ' '.join(strings)

    def as_command(self, only_used=False, only_changed=False, strip_pwd=True, strip_python_path=False) -> str:
        import sys
        command = sys.argv[0]
        command = util.simplify_path(command, strip_pwd_=strip_pwd, strip_python_path_=strip_python_path)
        return f'{command} {self.as_args(only_used=only_used, only_changed=only_changed)}'

    def get_launch_command(self, strip_pwd=True, strip_python_path=False):
        import sys
        command = ' '.join(sys.argv)
        command = util.simplify_path(command, strip_pwd_=strip_pwd, strip_python_path_=strip_python_path)
        return command

    @classmethod
    def from_dict(cls, kwargs):
        return cls(**kwargs)

    @classmethod
    def get_parser(cls):
        parser = argparse.ArgumentParser()
        for k, f in cls._fields_public_.items():
            flag = f'--{k.replace("_", "-")}'

            if f._type is bool:
                parser.add_argument(
                    flag,
                    default=f._default,
                    action=('store_false' if f._default else 'store_true')
                )
            else:
                parser.add_argument(
                    flag,
                    required=f._required,
                    type=f._cast if f._cast else (f._type if f._type is not _NONE else None),
                    default=f._default,
                    choices=f._choices,
                )
        return parser

    @classmethod
    def from_system_args(cls):
        args = cls.get_parser().parse_args()
        return cls.from_dict(vars(args))

    def print_reproduce_info(self):
        util.print_separator('REPRODUCE EXPERIMENT:')
        print('[COMMAND MINIMAL]:')
        print('    $', self.as_command(only_used=True, only_changed=True))
        print()
        print('[COMMAND USED]:')
        print('    $', self.as_command(only_used=True, only_changed=False))
        print()
        print('[COMMAND ALL]:')
        print('    $', self.as_command(only_used=False, only_changed=False))
        print()

    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join(f"{k}={getattr(self, k).__repr__()}" for k in self._fields_public_)})'


# ========================================================================= \#
# END                                                                       \#
# ========================================================================= \#


if __name__ == '__main__':

    class TestArgs(Args):
        a: int    = field(default=2)
        b: int    = field(default=3)
        name: str = field(default='unnamed-experiment')

        @computed
        def prod(self):
            return self.a * self.b

        @property
        def sum(self):
            return self.a + self.b

    print(TestArgs().prod)
    print(TestArgs().sum)
    print(TestArgs().get_dict_computed())


