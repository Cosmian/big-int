#[macro_export]

macro_rules! impl_ops_trait {
    ($Self: ty,
     $Other: ty,
     $Trait: ident { $method: ident },
     $TraitAssign: ident { $method_assign: ident },
     $function: ident,
     $function_assign: ident) => {
        // impl Self ops Other
        impl std::ops::$Trait<$Other> for $Self {
            type Output = $Self;

            #[inline]
            fn $method(self, other: $Other) -> Self::Output {
                std::ops::$Trait::$method(&self, &other)
            }
        }

        // impl &Self ops Other
        impl<'a> std::ops::$Trait<$Other> for &'a $Self {
            type Output = $Self;

            #[inline]
            fn $method(self, other: $Other) -> Self::Output {
                std::ops::$Trait::$method(self, &other)
            }
        }

        // impl Self ops &Other
        impl<'a> std::ops::$Trait<&'a $Other> for $Self {
            type Output = $Self;

            #[inline]
            fn $method(self, other: &'a $Other) -> Self::Output {
                std::ops::$Trait::$method(&self, other)
            }
        }

        // impl &Self ops &Other
        impl<'a, 'b> std::ops::$Trait<&'b $Other> for &'a $Self {
            type Output = $Self;

            #[inline]
            fn $method(self, other: &'b $Other) -> Self::Output {
                self.$function(other)
            }
        }

        // impl Self ops= Other
        impl std::ops::$TraitAssign<$Other> for $Self {
            #[inline]
            fn $method_assign(&mut self, other: $Other) {
                std::ops::$TraitAssign::$method_assign(self, &other);
            }
        }

        // impl Self ops= &Other
        impl<'a> std::ops::$TraitAssign<&'a $Other> for $Self {
            #[inline]
            fn $method_assign(&mut self, other: &'a $Other) {
                self.$function_assign(other);
            }
        }
    };
}
