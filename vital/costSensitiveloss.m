function Y = costSensitiveloss( X,c,dzdy )

sz = [size(X,1) size(X,2) size(X,3) size(X,4)] ;

c = c - 1 ;

if numel(c) == sz(4)
  c = reshape(c, [1 1 1 sz(4)]) ;
  c = repmat(c, [sz(1) sz(2)]) ;
else  
  sz_ = [size(c,1) size(c,2) size(c,3) size(c,4)] ;
  assert(isequal(sz_, [sz(1) sz(2) 1 sz(4)])) ;
end

c_ = 0:numel(c)-1 ;
c_ = 1 + ...
  mod(c_, sz(1)*sz(2)) + ...
  (sz(1)*sz(2)) * c(:)' + ...
  (sz(1)*sz(2)*sz(3)) * floor(c_/(sz(1)*sz(2))) ;

Xmax = max(X,[],3) ;
ex = exp(bsxfun(@minus, X, Xmax)) ;

n = sz(1)*sz(2) ;
if nargin <= 2
  t = Xmax + log(sum(ex,3)) - reshape(X(c_), [sz(1:2) 1 sz(4)]) ;
  Y = sum(t(:)) / n ;
else
  Y = bsxfun(@rdivide, ex, sum(ex,3)) ;
  Y(c_) = Y(c_) - 1;
  Y = Y * (dzdy / n) ;
end


end

